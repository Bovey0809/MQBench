import json
import os


import onnx
import numpy as np
from onnx import numpy_helper

from mqbench.utils.logger import logger
from mqbench.deploy.common import (
    update_inp2node_out2node,
    prepare_initializer,
    prepare_data_nnie,
    OnnxPreprocess,
    get_constant_inputs
)


class NNIE_process(object):
    def gen_gfpq_param_file(self, graph, clip_val):
        nnie_exclude_layer_type = ['Flatten', 'Relu', 'PRelu', 'Sigmoid', 'Reshape',
                                   'Softmax', 'CaffeSoftmax', 'Clip', 'GlobalAveragePool', 'Mul']
        interp_layer_cnt = 0
        gfpq_param_dict = {}
        for idx, node in enumerate(graph.node):
            # We can not support NNIE group conv.
            # Group conv need group-size input params.
            if node.op_type == 'Conv' and node.attribute[1].i != 1:
                continue

            layer_input_tensor = []
            for in_tensor in node.input:
                if in_tensor in clip_val:
                    clip_value = clip_val[in_tensor]
                    layer_input_tensor.append(float(clip_value))
                # Upsample layer only reserve one input.
                if node.op_type in ['Upsample', 'DynamicUpsample']:
                    break

            if node.op_type not in nnie_exclude_layer_type and len(layer_input_tensor) > 0:
                gfpq_param_dict[node.name] = layer_input_tensor

            # Upsample ---> Upsample + Permute in NNIE.
            if node.op_type in ['Upsample', 'DynamicUpsample']:
                interp_layer_name = node.name
                gfpq_param_dict[interp_layer_name + '_permute_' + str(interp_layer_cnt)] = gfpq_param_dict[interp_layer_name]
                interp_layer_cnt += 1
        return gfpq_param_dict

    def remove_fakequantize_and_collect_params(self, onnx_path, model_name):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data_nnie(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.replace_resize_op_with_upsample(graph, out2node)
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        nodes_to_be_removed = []
        clip_ranges = {}
        processed_outputs = set()  # 用于跟踪已处理的输出
        processed_nodes = set()    # 用于跟踪节点索引而不是节点本身

        for node in graph.node:
            if node.op_type == 'QuantizeLinear':
                # 获取量化节点的所有输出连接
                quant_outputs = node.output
                for quant_output in quant_outputs:
                    if quant_output in processed_outputs:
                        continue
                    
                    next_nodes = inp2node[quant_output]
                    # 获取原始输入
                    original_input = node.input[0]
                    # 记录clip range
                    clip_ranges[original_input] = name2data[node.input[1]]

                    # 处理所有使用这个输出的节点
                    for next_node, idx in next_nodes:
                        if next_node.op_type == 'Cast':
                            # 如果是Cast节点，需要处理Cast节点的所有输出连接
                            cast_outputs = next_node.output
                            for cast_output in cast_outputs:
                                if cast_output in processed_outputs:
                                    continue
                                
                                cast_next_nodes = inp2node[cast_output]
                                # 将所有使用Cast输出的节点重新连接到原始输入
                                for cast_next_node, cast_next_idx in cast_next_nodes:
                                    cast_next_node.input[cast_next_idx] = original_input
                                processed_outputs.add(cast_output)
                            
                            # 将Cast节点加入待删除列表（使用节点名称作为标识）
                            node_name = next_node.name if hasattr(next_node, 'name') else str(id(next_node))
                            if node_name not in processed_nodes:
                                nodes_to_be_removed.append(next_node)
                                processed_nodes.add(node_name)
                        else:
                            # 直接连接到原始输入
                            next_node.input[idx] = original_input
                    
                    processed_outputs.add(quant_output)
                
                # 将量化节点及其常量输入加入待删除列表
                node_name = node.name if hasattr(node, 'name') else str(id(node))
                if node_name not in processed_nodes:
                    nodes_to_be_removed.append(node)
                    processed_nodes.add(node_name)
                    
                    # 添加常量输入节点
                    constant_inputs = get_constant_inputs(node, out2node)
                    for const_node in constant_inputs:
                        const_name = const_node.name if hasattr(const_node, 'name') else str(id(const_node))
                        if const_name not in processed_nodes:
                            nodes_to_be_removed.append(const_node)
                            processed_nodes.add(const_name)

        # 删除所有标记的节点（保持原始顺序）
        for node in nodes_to_be_removed:
            if node in graph.node:
                graph.node.remove(node)

        gfpq_param_dict = self.gen_gfpq_param_file(graph, clip_ranges)

        output_path = os.path.dirname(onnx_path)
        gfpq_param_file = os.path.join(output_path, '{}_gfpq_param_dict.json'.format(model_name))
        with open(gfpq_param_file, 'w') as f:
            json.dump({"nnie": {"gfpq_param_dict": gfpq_param_dict}}, f, indent=4)
        onnx_filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(model_name))
        onnx.save(model, onnx_filename)
        logger.info("Finish deploy process.")


remove_fakequantize_and_collect_params_nnie = NNIE_process().remove_fakequantize_and_collect_params