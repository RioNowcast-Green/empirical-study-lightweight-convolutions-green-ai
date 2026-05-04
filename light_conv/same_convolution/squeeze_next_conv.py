from torch import nn


# Source: https://github.com/osmr/pytorchcv/blob/master/pytorchcv/models/squeezenext.py
# Note: This code contains some modifications compared to the original implementation.
class SqueezeNextConv(nn.Module):
    """
    SqueezeNext unit.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True
    ):
        super(SqueezeNextConv, self).__init__()

        if padding == "same":
            padding = kernel_size // 2
        elif padding == "valid":
            padding = 0

        # reduction_ch = (in_channels // 2)
        # reduction_ch = 1 if reduction_ch == 0 else reduction_ch
        reduction_ch = 1

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=reduction_ch,
            kernel_size=1,
            stride=stride,
            bias=bias,
        )
        self.conv3 = nn.Conv2d(
            in_channels=reduction_ch,
            out_channels=1,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, padding),
            bias=bias,
        )
        self.conv4 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(padding, 0),
            bias=bias,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
