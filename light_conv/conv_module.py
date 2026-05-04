import torch.nn as nn
from light_conv.same_convolution.depthwise_separable_conv import DepthwiseSeparableConv
from light_conv.same_convolution.squeeze_next_conv import SqueezeNextConv
from light_conv.same_convolution.tied_block_conv import TiedBlockConv
from light_conv.same_convolution.global_conv import GlobalConv
from light_conv.same_convolution.dws_bottleneck_conv import DWSBottleneckConv
from light_conv.same_convolution.fire_conv import FireConv


def ConvModule(type, **conv_params) -> nn.Module:
    conv_list = {
        "standard": nn.Conv2d,
        "dws": DepthwiseSeparableConv,
        "sqnxt": SqueezeNextConv,
        "tied": TiedBlockConv,
        "global": GlobalConv,
        "dwsb": DWSBottleneckConv,
        "fire": FireConv,
    }

    try:
        return conv_list[type](**conv_params)
    except KeyError:
        raise ValueError(f"Unsupported convolution type: {type}")
