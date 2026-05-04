import torch
import torch.nn as nn


# Source: https://github.com/frank-xwang/TBC-TiedBlockConvolution/blob/main/TiedBlockConv.py
# Note: This code contains some modifications compared to the original implementation.
class TiedBlockConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        B=60,
        groups=1,
    ):
        super(TiedBlockConv, self).__init__()
        self.B = B
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.split_by_B = (in_channels >= B) and (out_channels >= B)
        self.out_channels_remainder = out_channels % B if self.split_by_B else 0
        self.in_channels_remainder = in_channels % B if self.split_by_B else 0
        self.in_channels_to_work = in_channels - self.in_channels_remainder
        self.out_channels_to_work = out_channels - self.out_channels_remainder

        if padding == "same":
            padding = kernel_size // 2
        elif padding == "valid":
            padding = 0

        self.padding = padding
        self.conv = nn.Conv2d(
            (
                self.in_channels_to_work // self.B
                if self.split_by_B
                else self.in_channels_to_work
            ),
            (
                self.out_channels_to_work // self.B
                if self.split_by_B
                else self.out_channels_to_work
            ),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        n, c, h, w = x.size()
        x_remainder = None

        if self.split_by_B:
            if self.in_channels_remainder != 0:
                x_remainder = x[:, self.in_channels_to_work :, :, :]
            x = x[:, : self.in_channels_to_work, :, :].contiguous()
            x = x.view(n * self.B, self.in_channels_to_work // self.B, h, w)
        else:
            x = x.contiguous()

        x = self.conv(x)

        if self.split_by_B:
            h_o = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
            w_o = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

            x = x.view(n, self.out_channels_to_work, h_o, w_o)

            if self.out_channels_remainder != 0:
                if self.out_channels_remainder < self.in_channels_remainder:
                    x = torch.cat(
                        [x, x_remainder[:, : self.out_channels_remainder, :h_o, :w_o]],
                        dim=1,
                    )
                elif self.out_channels_remainder == self.in_channels_remainder:
                    x = torch.cat([x, x_remainder[:, :, :h_o, :w_o]], dim=1)
                else:
                    if self.in_channels_remainder != 0:
                        channels_left = (
                            self.out_channels_remainder - self.in_channels_remainder
                        )
                        x = torch.cat([x, x_remainder[:, :, :h_o, :w_o]], dim=1)
                        x = torch.cat([x, x[:, :channels_left, :, :]], dim=1)
                    else:
                        x = torch.cat(
                            [x, x[:, : self.out_channels_remainder, :, :]], dim=1
                        )
        return x
