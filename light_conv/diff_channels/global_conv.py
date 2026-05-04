import torch.nn as nn


# Source: https://github.com/SConsul/Global_Convolutional_Network/blob/master/model/GCN.py
# Note: This code contains some modifications compared to the original implementation.
class GlobalConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, k=3, per_out_channels=1
    ):  # out_Channel=21 in paper
        super(GlobalConv, self).__init__()
        self.out_channels = out_channels

        self.conv_l1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=per_out_channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
        )
        self.conv_l2 = nn.Conv2d(
            in_channels=per_out_channels,
            out_channels=per_out_channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
        )
        self.conv_r1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=per_out_channels,
            kernel_size=(1, k),
            padding=(k // 2, 0),
        )
        self.conv_r2 = nn.Conv2d(
            in_channels=per_out_channels,
            out_channels=per_out_channels,
            kernel_size=(k, 1),
            padding=(0, k // 2),
        )

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        x = x.repeat(1, self.out_channels, 1, 1)  # [N, C*reps, H, W]

        return x
