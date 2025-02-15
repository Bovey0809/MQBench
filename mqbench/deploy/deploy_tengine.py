import os
from collections import OrderedDict

from ..utils.logger import logger
from .deploy_linear import (
    LinearQuantizer_process,
    ALL_FAKEQUANTIZER,
    PERCHANNEL_FAKEQUANTIZER,
    PERTENSOR_FAKEQUANTIZER
)
from .common import (
    update_inp2node_out2node,
    prepare_initializer,
    prepare_data,
    OnnxPreprocess,
    get_constant_inputs,
    parse_attrs
)

import onnx
from onnx import numpy_helper
try:
    from onnxsim import simplify
except ModuleNotFoundError:
    logger.warn('onnxsim not found, if you want to use deploy_tengine, please install it.')


def get_dequant_node(node, inp2node):
    dequant_node = inp2node[node.output[0]][0][0]
    if dequant_node.op_type == 'Clip':
        dequant_node = inp2node[dequant_node.output[0]][0][0]
    assert dequant_node.op_type == 'DequantizeLinear', "This is not correct fakequant node!"
    return dequant_node

class Tengine_process(LinearQuantizer_process):

    @staticmethod
    def get_constant(node: onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t).tolist()

    def remove_fakequantize_and_collect_params(self, onnx_path, model_name, qmin_max_dict):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        quant_params = OrderedDict()
        nodes_to_be_removed = []
        for node in graph.node:
            if node.op_type in ALL_FAKEQUANTIZER:
                next_node = inp2node[node.output[0]][0][0]
                if next_node.op_type == 'Clip' and inp2node[next_node.output[0]][0][0].op_type == 'DequantizeLinear':
                    nodes_to_be_removed.append(next_node)
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))

            if node.op_type == 'QuantizeLinear' and 'axis' in parse_attrs(node.attribute):
                # fake quantize for weights, suppose per-channel quantize only for weight
                qmin, qmax = qmin_max_dict[node.name]
                dequant_node = get_dequant_node(node, inp2node)
                redundant_nodes = self.deal_with_weight_fakequant(node, dequant_node, out2node, inp2node, named_initializer)
                nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node, dequant_node, name2data, inp2node, named_initializer, qmin, qmax)
            elif node.op_type == 'QuantizeLinear' and 'axis' not in parse_attrs(node.attribute):
                qmin, qmax = qmin_max_dict[node.name]
                dequant_node = get_dequant_node(node, inp2node)
                if dequant_node.output[0] not in inp2node:
                    assert dequant_node.output[0] in [x.name for x in graph.output]
                    inp2node[dequant_node.output[0]] = []

                next_nodes = inp2node[dequant_node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, dequant_node, out2node, inp2node, named_initializer)
                    tensor_name, scale, zero_point = self.parse_qparams(node, name2data)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, dequant_node, name2data, inp2node, named_initializer, qmin, qmax)
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, dequant_node, inp2node)
                    tensor_name, scale, zero_point = self.parse_qparams(node, name2data)
                    for out in graph.output:
                        if out.name == dequant_node.output[0]:
                            out.name = tensor_name

                    quant_params[tensor_name] = [
                        float(scale),
                        int(zero_point)
                    ]

                    # detect fusion for tengine graph
                    # since tengine convert tool will optimize graph
                    # by fusing conv+relu, conv+relu6
                    # ref: https://github.com/OAID/Tengine/blob/cdb4ccf77c04a0a771ec6a43631b9d25acd2bae1/tools/convert_tool/utils/graph_optimizer/graph_opt.cpp#L941
                    pre_node = out2node.get(tensor_name, None)
                    if pre_node and pre_node.op_type in {"Clip", "Relu"}:
                        # suppose onnx version be 11
                        # for relu6
                        if pre_node.op_type == "Clip" and \
                            not (self.get_constant(out2node[pre_node.input[1]]) == 0 and
                                 self.get_constant(out2node[pre_node.input[2]]) == 6):
                            continue

                        conv_node = out2node[pre_node.input[0]]
                        if conv_node.op_type == "Conv":
                            conv_tensor_name = conv_node.output[0]
                            quant_params[conv_tensor_name] = quant_params[tensor_name]

        for node in nodes_to_be_removed:
            if node in graph.node:
                graph.node.remove(node)
        named_initializer = prepare_initializer(graph)
        for name, initial_data in named_initializer.items():
            if name in (out2node.keys() | inp2node.keys()):
                continue
            graph.initializer.remove(initial_data)

        # TODO: softmax
        quant_params = self.post_process_clip_ranges(quant_params, graph, inp2node)
        output_path = os.path.dirname(onnx_path)
        context_filename = os.path.join(output_path, f"{model_name}_for_tengine.scale")
        with open(context_filename, "w") as f:
            for name, value in quant_params.items():
                scale, zero_point = value
                f.write(f"{name} {scale} {zero_point}\n")

        model_opt, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"

        onnx_filename = os.path.join(output_path, f"{model_name}_for_tengine.onnx")
        onnx.save(model_opt, onnx_filename)

        logger.info("Finish deploy process.")


remove_fakequantize_and_collect_params_tengine = Tengine_process().remove_fakequantize_and_collect_params
