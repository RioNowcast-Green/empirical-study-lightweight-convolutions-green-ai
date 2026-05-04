import torch.nn as nn


# Based idea in "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
class DWSBottleneckConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True
    ):
        super(DWSBottleneckConv, self).__init__()
        self.pointwise_rec = nn.Conv2d(
            in_channels, 1, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.depthwise = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=1,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            1, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(self.pointwise_rec(x)))
