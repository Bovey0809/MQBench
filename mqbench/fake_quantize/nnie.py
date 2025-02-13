import torch

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import no_jit_trace
from torch.onnx import (
    symbolic_helper,
)
from torch.onnx import register_custom_op_symbolic
class NNIEFakeQuantize(QuantizeBase):
    def __init__(self, observer, **observer_kwargs):
        super(NNIEFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('data_max', torch.tensor(float('-inf')))
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int32))

    def forward(self, X):
        with no_jit_trace():
            if self.observer_enabled[0] == 1:
                self.activation_post_process(X.detach())
                data_max = torch.max(-self.activation_post_process.min_val, self.activation_post_process.max_val)
                self.data_max = torch.max(data_max, self.data_max)
                self.scale = self.data_max.clone().detach()
                self.zero_point = torch.zeros_like(self.scale, dtype=torch.int32)
        if self.fake_quant_enabled[0] == 1:
            X = NNIEQuantizeFunc.apply(X, self.data_max, self.scale, self.zero_point)
        return X


class NNIEQuantizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_max, scale, zero_point):
        # 保存用于反向传播的值
        ctx.save_for_backward(x, data_max)
        
        z = (16 * torch.log2(data_max.double())).round() - 127
        x = x.double()
        pos_idx = x > 2 ** ((z - 16) / 16)
        neg_idx = x < - 2 ** ((z + 1 - 16) / 16)
        zero_idx = (x >= - 2 ** ((z + 1 - 16) / 16)) & (x < 2 ** ((z - 16) / 16))
        x[zero_idx] = 0
        x[pos_idx] = 2 ** ((torch.clamp(torch.round(16 * torch.log2(x[pos_idx]) - z), 0, 127) + z) / 16)
        x[neg_idx] = - 2 ** ((torch.clamp(torch.round(16 * torch.log2(-x[neg_idx]) - z), 1, 127) + z) / 16)
        x = x.float()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, data_max = ctx.saved_tensors
        grad_input = grad_output.clone()
        # 返回与forward参数数量相同的梯度
        return grad_input, None, None, None

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "v")
    def symbolic(g, x, data_max, scale, zero_point):
        # 使用scale和zero_point进行ONNX导出
        return g.op("QuantizeLinear", x, scale, zero_point)