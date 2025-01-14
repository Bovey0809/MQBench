import torch
import torchvision
import unittest

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization


class TestFakeQuantize(unittest.TestCase):
    def test_fixed_fake_quantize(self):
        model_to_quantize = torchvision.models.resnet18(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_fixed')

    def test_learnable_fake_quantize(self):
        model_to_quantize = torchvision.models.resnet18(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_lsq')

    def test_nnie_fake_quantize(self):
        model_to_quantize = torchvision.models.resnet18(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'NNIEFakeQuantize',
            'a_fakequantize': 'NNIEFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.NNIE, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.NNIE, {'x': [1, 3, 224, 224]}, model_name='resnet18_nnie')

    def test_dorefa_fake_quantize(self):
        model_to_quantize = torchvision.models.resnet18(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'DoReFaFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize'
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_dorefa_trt')

    def test_pact_fake_quantize(self):
        model_to_quantize = torchvision.models.resnet18(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'PACTFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_pact_trt')

    def test_dsq_fake_quantize(self):
        model_to_quantize = torchvision.models.resnet18(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'ClipStdObserver',
            'a_observer': 'ClipStdObserver',
            'w_fakequantize': 'DSQFakeQuantize',
            'a_fakequantize': 'DSQFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_dsq_trt')