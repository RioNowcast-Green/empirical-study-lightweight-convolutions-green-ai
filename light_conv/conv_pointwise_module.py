from functools import partial
import torch.nn as nn


def ConvPointWiseModule(type, **conv_params) -> nn.Module:
    conv_list = {
        "standard": partial(nn.Conv2d, kernel_size=1),
    }

    try:
        return conv_list[type](**conv_params)
    except KeyError:
        raise ValueError(f"Unsupported convolution type: {type}")
