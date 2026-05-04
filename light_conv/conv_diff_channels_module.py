import torch.nn as nn

from light_conv.conv_module import ConvModule

from light_conv.diff_channels.global_conv import GlobalConv
from light_conv.diff_channels.fire_conv import FireConv


def ConvDiffChannelsModule(type, **conv_params) -> nn.Module:
    conv_list = {
        "standard": nn.Conv2d,
        "global": GlobalConv,
        "fire": FireConv,
    }

    conv = conv_list.get(type, None)
    if conv is None or type == "standard":
        return ConvModule(type, **conv_params)
    diff_channels_params = {
        k: conv_params[k] for k in ("in_channels", "out_channels") if k in conv_params
    }
    return conv(**diff_channels_params)
