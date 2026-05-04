import torch.nn as nn
import torch.nn.functional as F


# Source: https://github.com/SConsul/Global_Convolutional_Network/blob/master/model/GCN.py
# Note: This code contains some modifications compared to the original implementation.
class GlobalConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):  # out_Channel=21 in paper
        super(GlobalConv, self).__init__()

        if padding == "same":
            padding = kernel_size // 2
        elif padding == "valid":
            padding = 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        per_out_channels = 1

        self.conv_l1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=per_out_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=bias,
        )
        self.conv_l2 = nn.Conv2d(
            in_channels=per_out_channels,
            out_channels=per_out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=bias,
        )
        self.conv_r1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=per_out_channels,
            kernel_size=(1, kernel_size),
            padding=(kernel_size // 2, 0),
            bias=bias,
        )
        self.conv_r2 = nn.Conv2d(
            in_channels=per_out_channels,
            out_channels=per_out_channels,
            kernel_size=(kernel_size, 1),
            padding=(0, kernel_size // 2),
            bias=bias,
        )

    def forward(self, x):
        b, c, h, w = x.size()
        h_o = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_o = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        x = x.repeat(1, self.out_channels, 1, 1)  # [N, C*reps, H, W]

        if x.shape[2] >= h_o and x.shape[3] >= w_o:
            return x[:, :, :h_o, :w_o]
        x = F.interpolate(x, size=(h_o, w_o), mode="nearest")

        return x
