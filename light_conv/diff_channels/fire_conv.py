from math import ceil
import torch
import torch.nn as nn


# Source: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
# Note: This code contains some modifications compared to the original implementation.
class FireConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_planes: int = 1,
        expand1x1_planes: int = 1,
        expand3x3_planes: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.inplanes = in_channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)

        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)

        if x.shape[1] >= self.out_channels:
            x = x[:, : self.out_channels]
        else:
            reps = ceil(self.out_channels / x.shape[1])
            x = x.repeat(1, reps, 1, 1)  # [N, C*reps, H, W]
            x = x[:, : self.out_channels]

        return x
