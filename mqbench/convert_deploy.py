import os.path as osp

import torch
from torch.fx import GraphModule

import mqbench.custom_symbolic_opset  # noqa: F401
import mqbench.fusion_method          # noqa: F401
from mqbench.prepare_by_platform import BackendType
from mqbench.utils import deepcopy_graphmodule
from mqbench.utils.logger import logger
from mqbench.utils.registry import (
    BACKEND_DEPLOY_FUNCTION,
    register_deploy_function,
    FUSED_MODULE_CONVERT_FUNCTION
)
from mqbench.deploy import (
    remove_fakequantize_and_collect_params_nnie,
    remove_fakequantize_and_collect_params,
    replace_fakequantize_and_collect_params_openvino,
    remove_fakequantize_and_collect_params_tengine,
    remove_fakequantize_and_collect_params_stpu,
    ONNXQLinearPass, ONNXQNNPass
)
import onnx
from onnxsim import simplify
from mqbench.deploy.common import (
    parse_attrs
)
from torch.fx import subgraph_rewriter
import operator
import inspect

from mqbench.utils.utils import getitem2node
from mqbench.fake_quantize.nnie import NNIEFakeQuantize

__all__ = ['convert_deploy']
qmin_max_dict = {}
@register_deploy_function(BackendType.STPU)
@register_deploy_function(BackendType.Tengine_u8)
@register_deploy_function(BackendType.PPLCUDA)
@register_deploy_function(BackendType.ONNX_QNN)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
@register_deploy_function(BackendType.Vitis)
@register_deploy_function(BackendType.OPENVINO)
@register_deploy_function(BackendType.QDQ)
def convert_merge_bn(model: GraphModule, **kwargs):
    logger.info("Merge BN for deploy.")
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    for node in nodes:
        if node.op == 'call_module':
            if type(modules[node.target]) in FUSED_MODULE_CONVERT_FUNCTION:
                FUSED_MODULE_CONVERT_FUNCTION[type(modules[node.target])](model, node)


@register_deploy_function(BackendType.STPU)
@register_deploy_function(BackendType.Academic_NLP)
@register_deploy_function(BackendType.Tensorrt_NLP)
@register_deploy_function(BackendType.Tengine_u8)
@register_deploy_function(BackendType.PPLCUDA)
@register_deploy_function(BackendType.ONNX_QNN)
@register_deploy_function(BackendType.Academic)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
@register_deploy_function(BackendType.Vitis)
@register_deploy_function(BackendType.OPENVINO)
@register_deploy_function(BackendType.QDQ)
def convert_onnx(model: GraphModule, input_shape_dict, dummy_input, onnx_model_path, **kwargs):
    logger.info("Export to onnx.")
    output_names = kwargs.get('output_names', [])
    dynamic_axes = kwargs.get('dynamic_axes', {})
    input_names = kwargs.get('input_names', [])
    
    # First extract image tensor from input dict
    # model = extract_input_tensors(model)
    
    # Then handle other dict operations
    # model = remove_dict_operations(model)
    
    # Convert dummy_input from dict to tensor if needed
    if isinstance(dummy_input, dict) and 'image' in dummy_input:
        dummy_input = dummy_input['image']
    
    if dummy_input is None and input_shape_dict:
        device = next(model.parameters()).device
        if 'image' in input_shape_dict:
            dummy_input = torch.rand(input_shape_dict['image']).to(device)
        else:
            # Original dictionary input creation
            dummy_input = {name: torch.rand(shape).to(device) 
                         for name, shape in input_shape_dict.items()}
            input_names = list(dummy_input.keys())
            dummy_input = tuple(dummy_input.values())
    
    # Per-channel QuantizeLinear and DequantizeLinear is supported since opset 13
    opset_version = 13 if kwargs.get('deploy_to_qlinear', False) else 13
    print("=== 量化参数验证 ===")
    for name, param in model.named_parameters():
        if 'scale' in name or 'zero_point' in name:
            print(f"参数存在: {name} | 值: {param.data}")
    
    # 添加默认量化参数
    # model = add_default_quant_params(model)
    
    # 验证量化节点
    # validation_passed = validate_quantize_nodes(model)
    # if not validation_passed:
        # raise RuntimeError("QuantizeLinear节点验证失败，请检查上述输出")
    
    # split_and_debug_model(model, dummy_input)
    
    with torch.no_grad():
        # 确保所有量化节点都有必要的参数
        for name, module in model.named_modules():
            if isinstance(module, NNIEFakeQuantize):
                if not hasattr(module, 'scale'):
                    module.register_buffer('scale', module.data_max.clone().detach())
                if not hasattr(module, 'zero_point'):
                    module.register_buffer('zero_point', torch.zeros_like(module.scale, dtype=torch.int32))
        
        torch.onnx.export(model, dummy_input, onnx_model_path,
                         input_names=input_names,
                         output_names=output_names,
                         opset_version=opset_version,
                         dynamic_axes=dynamic_axes,
                         do_constant_folding=True,
                         enable_onnx_checker=False,
                         custom_opsets={'' : opset_version})
        
        onnx_model = onnx.load(onnx_model_path)
        graph = onnx_model.graph
        for node in graph.node:
            if len(node.attribute) > 1:
                qparams = parse_attrs(node.attribute)
                if 'quant_max' in qparams:
                    qmin_max_dict[node.name] = (qparams['quant_min'], qparams['quant_max'])
                    new_attributes = []
                    for attr in node.attribute:
                        if attr.name not in ["quant_min", "quant_max"]:
                            new_attributes.append(attr)
                    node.ClearField("attribute")
                    node.attribute.extend(new_attributes)
        onnx.save(onnx_model, onnx_model_path)
        try:
            logger.info("simplify model.")
            onnx_model = onnx.load(onnx_model_path)
            onnx_model_simplified, check = simplify(onnx_model)
            onnx.save(onnx_model_simplified, onnx_model_path)
        except Exception as e:
            logger.info("simplify model fail.")


@register_deploy_function(BackendType.Tensorrt)
def convert_onnx_qlinear(model: GraphModule, onnx_model_path, model_name, **kwargs):
    if kwargs.get('deploy_to_qlinear', False):
        logger.info("Convert to ONNX QLinear.")
        ONNXQLinearPass(onnx_model_path).run()


@register_deploy_function(BackendType.NNIE)
def deploy_qparams_nnie(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for NNIE.")
    remove_fakequantize_and_collect_params_nnie(onnx_model_path, model_name)


@register_deploy_function(BackendType.OPENVINO)
def deploy_qparams_openvino(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for OPENVINO.")
    replace_fakequantize_and_collect_params_openvino(onnx_model_path, model_name, qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.Tensorrt)
def deploy_qparams_tensorrt(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for TensorRT.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='tensorrt', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.Vitis)
def deploy_qparams_vitis(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Vitis-DPU.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='vitis', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.SNPE)
def deploy_qparams_snpe(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for SNPE.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='snpe', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.PPLW8A16)
def deploy_qparams_pplw8a16(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for PPLW8A16.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='ppl', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.ONNX_QNN)
def deploy_qparams_tvm(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Convert to ONNX QNN.")
    ONNXQNNPass(onnx_model_path).run(model_name, qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.PPLCUDA)
def deploy_qparams_ppl_cuda(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for PPL-CUDA.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='ppl-cuda', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.Tengine_u8)
def deploy_qparams_tengine(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Tengine.")
    remove_fakequantize_and_collect_params_tengine(onnx_model_path, model_name, qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.STPU)
def deploy_qparams_stpu(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for STPU.")
    remove_fakequantize_and_collect_params_stpu(onnx_model_path, model_name, qmin_max_dict = qmin_max_dict)


def remove_dict_operations(model: GraphModule):
    # 获取getitem到源节点的映射
    getitem_map = getitem2node(model)
    
    # 创建新图
    new_graph = torch.fx.Graph()
    value_map = {}
    
    for node in model.graph.nodes:
        # 替换getitem节点
        if node.target == operator.getitem and node in getitem_map:
            src_node = getitem_map[node]
            value_map[node] = value_map[src_node]
            continue
            
        # 替换dict.update操作
        if node.target == dict.update:
            # 将字典更新转换为元组赋值
            new_args = _fix_succ_recursivly(node.args, node.args[0], node.args[1])
            new_node = new_graph.call_function(tuple, args=(new_args[1],))
            value_map[node] = new_node
            continue
            
        # 复制其他节点
        new_node = new_graph.node_copy(node, lambda n: value_map[n])
        value_map[node] = new_node
    
    return GraphModule(model, new_graph)

def extract_input_tensors(model: GraphModule):
    """修复版：保留所有字典输入项"""
    new_graph = torch.fx.Graph()
    value_map = {}
    
    # 收集所有字典输入项
    input_node = None
    input_getitems = {}
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            input_node = node
        elif node.op == 'call_function' and node.target == operator.getitem:
            if node.args[0] == input_node:
                key = node.args[1]
                input_getitems[key] = node
                
    # 创建新placeholder并建立映射
    new_placeholder = new_graph.placeholder('input_dict')
    value_map[input_node] = new_placeholder
    
    # 处理所有getitem节点
    for key, getitem_node in input_getitems.items():
        # 创建新的getitem节点
        new_getitem = new_graph.call_function(
            operator.getitem,
            args=(new_placeholder, key)
        )
        value_map[getitem_node] = new_getitem
    
    # 复制其他节点
    for node in model.graph.nodes:
        if node in [input_node] + list(input_getitems.values()):
            continue
        new_node = new_graph.node_copy(node, lambda n: value_map[n])
        value_map[node] = new_node
    
    return GraphModule(model, new_graph)

def split_and_debug_model(model: GraphModule, dummy_input):
    # Split the model at different stages
    nodes = list(model.graph.nodes)
    
    # Create progressive submodules
    for split_idx in range(1, len(nodes), 1):  # Check every single node
        try:
            # Create subgraph up to current node
            subgraph = torch.fx.Graph()
            node_map = {}
            
            for node in nodes[:split_idx]:
                new_node = subgraph.node_copy(node, lambda n: node_map[n])
                node_map[node] = new_node
            
            # Create output node
            subgraph.output(node_map[nodes[split_idx-1]])
            
            # Create submodule
            submodule = GraphModule(model, subgraph)
            
            # Special handling for quantization modules
            if 'post_act_fake_quantizer' in nodes[split_idx-1].target:
                # Quantizers expect single tensor input
                test_input = dummy_input[0] if isinstance(dummy_input, tuple) else dummy_input
                sub_out = submodule(test_input)
                print(f"Quantizer node inputs:")
                for arg in nodes[split_idx-1].args:
                    print(f"  {arg}")  # 应显示scale/zero_point参数来源
                print(f"Quantizer module params:")
                print(dict(submodule.named_parameters()))  # 检查参数是否存在
            else:
                sub_out = submodule(dummy_input)
                
            print(f"Passed nodes 0-{split_idx}: {nodes[split_idx-1].op} {nodes[split_idx-1].target}")
            
            # Test ONNX export
            torch.onnx.export(
                submodule,
                dummy_input,
                f"debug_{split_idx}.onnx",
                input_names=["input"],
                opset_version=13
            )
            print(f"Successfully exported up to node {split_idx}\n")
            
        except Exception as e:
            print(f"Failed at node {split_idx} ({nodes[split_idx-1].target}):")
            print(f"Error: {str(e)}")
            print("Last successful nodes:")
            for node in nodes[max(0, split_idx-5):split_idx]:
                print(f"  {node.op}: {node.target}")
            break

    # for node in model.graph.nodes:
    #     if 'post_act_fake_quantizer' in str(node.target):
    #         print(f"Quantizer node: {node.format_node()}")
    #         print(f"Input args: {node.args}")      # What's fed into this quantizer
    #         print(f"Input kwargs: {node.kwargs}\n") # Any keyword parameters

def validate_quantize_nodes(model: GraphModule):
    """验证所有QuantizeLinear节点的输入参数是否完整"""
    modules = dict(model.named_modules())
    quant_nodes = []
    
    # 收集所有QuantizeLinear节点
    for node in model.graph.nodes:
        if node.op == 'call_module' and 'post_act_fake_quantizer' in node.target:
            quant_nodes.append(node)
    
    # 检查每个量化节点
    for idx, node in enumerate(quant_nodes):
        print(f"\nChecking QuantizeLinear node {idx+1}/{len(quant_nodes)}:")
        print(f"Node name: {node.name}")
        print(f"Target module: {node.target}")
        
        # 检查模块参数
        module = modules[node.target]
        print("Module parameters:")
        for name, param in module.named_parameters():
            print(f"  {name}: {param.shape if param is not None else 'MISSING'}")
        
        # 检查输入参数
        print("\nInput arguments:")
        for i, arg in enumerate(node.args):
            arg_type = str(type(arg)).split("'")[1]
            print(f"Argument {i}: {arg} ({arg_type})")
            
            # 特别检查scale/zero_point来源
            if isinstance(arg, torch.fx.Node):
                print(f"  Source node: {arg.name} (op: {arg.op}, target: {arg.target})")
                if arg.op == 'get_attr':
                    attr_value = getattr(model, arg.target, None)
                    print(f"  Attribute value: {attr_value}")
        
        # 验证参数数量
        if len(node.args) < 2:
            print(f"❌ 错误：QuantizeLinear节点 {node.name} 只有 {len(node.args)} 个输入参数，至少需要2个")
        else:
            print("✅ 参数数量符合要求")
            
    return len(quant_nodes) > 0

def post_process_onnx(model: GraphModule):
    """将注释参数转换回实际值"""
    for node in model.onnx_model.graph.node:
        if node.op_type == 'Comment' and 'nnie_params' in node.attribute[0].s:
            params = parse_comment(node.attribute[0].s)  # 解析data_max
            # 找到对应的QLinearConv节点
            conv_node = find_consumer(node.output[0])
            # 替换虚拟参数为实际计算值
            conv_node.input[1] = calculate_real_scale(params['data_max'])
            conv_node.input[2] = create_zero_point()

def create_default_params(model):
    """创建默认量化参数常量"""
    if not hasattr(model, '_default_scale'):
        model._default_scale = model.graph.create_node(
            'call_function', 
            torch.quantize_per_tensor,
            args=(torch.tensor([1.0]),),
            kwargs={'dtype': torch.float32}
        )
    if not hasattr(model, '_default_zp'):
        model._default_zp = model.graph.create_node(
            'call_function',
            torch.quantize_per_tensor,
            args=(torch.tensor(0),),
            kwargs={'dtype': torch.uint8}
        )
    return model._default_scale, model._default_zp

def add_default_quant_params(model: GraphModule):
    """为每个QuantizeLinear节点添加默认的scale和zero_point参数"""
    new_graph = torch.fx.Graph()
    value_map = {}
    
    # 创建默认参数
    scale = new_graph.create_node(
        'call_function',
        torch.tensor,
        args=([1.0],),
        kwargs={'dtype': torch.float32, 'device': 'cpu'}
    )
    zero_point = new_graph.create_node(
        'call_function',
        torch.tensor,
        args=([0],),
        kwargs={'dtype': torch.int32, 'device': 'cpu'}
    )
    
    # 复制并修改图
    for node in model.graph.nodes:
        if node.op == 'call_module' and 'post_act_fake_quantizer' in node.target:
            # 为量化节点添加scale和zero_point参数
            new_args = list(node.args)
            if len(new_args) < 2:
                new_args.append(scale)  # 添加scale
            if len(new_args) < 3:
                new_args.append(zero_point)  # 添加zero_point
                
            # 创建新的量化节点
            new_node = new_graph.node_copy(node, lambda n: value_map[n])
            new_node.args = tuple(new_args)
            value_map[node] = new_node
        else:
            # 复制其他节点
            new_node = new_graph.node_copy(node, lambda n: value_map[n])
            value_map[node] = new_node
    
    return GraphModule(model, new_graph)

def convert_deploy(model: GraphModule, backend_type: BackendType,
                   input_shape_dict=None, dummy_input=None, output_path='./',
                   model_name='mqbench_qmodel', deploy_to_qlinear=False, **extra_kwargs):
    r"""Convert model to onnx model and quantization params depends on backend.

    Args:
        model (GraphModule): GraphModule prepared qat module.
        backend_type (BackendType): specific which backend should be converted to.
        input_shape_dict (dict): keys are model input name(should be forward function
                                 params name, values are list of tensor dims)
        output_path (str, optional): path to save convert results. Defaults to './'.
        model_name (str, optional): name of converted onnx model. Defaults to 'mqbench_qmodel'.

    >>> note on input_shape_dict:
        example: {'input_0': [1, 3, 224, 224]
                'input_1': [1, 3, 112, 112]
                }
        while forward function signature is like:
                def forward(self, input_0, input_1):
                    pass
    """
    kwargs = {
        'input_shape_dict': input_shape_dict,
        'dummy_input': dummy_input,
        'output_path': output_path,
        'model_name': model_name,
        'onnx_model_path': osp.join(output_path, '{}.onnx'.format(model_name)),
        'deploy_to_qlinear': deploy_to_qlinear,
    }
    # kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    for convert_function in BACKEND_DEPLOY_FUNCTION[backend_type]:
        convert_function(deploy_model, **kwargs)