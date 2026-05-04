import torch.nn as nn


# Source: https://github.com/EloyReulen/GA-SmaAt-GNet/blob/main/models/layers.py
class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True
    ):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
