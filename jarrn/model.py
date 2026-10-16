"""The two-branch JARRN model, extracted from the project's UNet implementation."""

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return self.weight * ((x - mean) / (std + self.eps)) + self.bias


class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rezero: bool):
        super().__init__()
        middle = in_channels // 4
        if middle < 1:
            raise ValueError("bottleneck requires at least four input channels")
        self.rezero = rezero
        if rezero:
            self.left = nn.Sequential(
                nn.LeakyReLU(0.2), nn.Conv2d(in_channels, middle, 1),
                nn.LeakyReLU(0.2), nn.Conv2d(middle, middle, 3, padding=1),
                nn.LeakyReLU(0.2), nn.Conv2d(middle, out_channels, 1),
            )
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.left = nn.Sequential(
                ChannelLayerNorm(in_channels), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels, middle, 1), ChannelLayerNorm(middle),
                nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(middle, middle, 3, padding=1),
                ChannelLayerNorm(middle), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(middle, out_channels, 1),
            )
        self.right = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = x if self.right is None else self.right(x)
        return residual + (self.alpha if self.rezero else 1.0) * self.left(x)


class StackedBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rezero: bool):
        super().__init__()
        self.block = nn.Sequential(
            BottleNeck(in_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
        )

    def forward(self, x):
        return self.block(x)


class FirstModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rezero: bool):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            StackedBottleNeck(out_channels, out_channels, rezero),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        y = self.block(x)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
        return torch.cat((x, y), dim=1)


class UNetModule(nn.Module):
    def __init__(self, middle: nn.Module, in_channels: int, middle_channels: int, rezero: bool):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, stride=2, padding=1),
            StackedBottleNeck(middle_channels, middle_channels, rezero),
            middle,
            StackedBottleNeck(2 * middle_channels, in_channels, rezero),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        y = self.block(x)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
        return torch.cat((x, y), dim=1)


class JARRN(nn.Module):
    """Visual depth branch followed by scale, offset and probability refinement.

    The default seven levels and channels match the source UNet model. Smaller
    configurations are useful for checking the training pipeline on a CPU.
    """

    def __init__(self, levels: int = 7, base_channels: int = 64, rezero: bool = False):
        super().__init__()
        if not 2 <= levels <= 7 or base_channels < 4:
            raise ValueError("levels must be 2..7 and base_channels at least 4")
        channels = [base_channels * x for x in (8, 8, 8, 8, 4, 2, 1)]

        def branch(inputs: int, outputs: int):
            layer = FirstModule(channels[1], channels[0], rezero)
            for i in range(2, levels):
                layer = UNetModule(layer, channels[i], channels[i - 1], rezero)
            i = levels - 1
            return nn.Sequential(
                nn.Conv2d(inputs, channels[i], 3, padding=1),
                StackedBottleNeck(channels[i], channels[i], rezero),
                layer,
                StackedBottleNeck(2 * channels[i], channels[i], rezero),
                nn.Conv2d(channels[i], outputs, 3, padding=1),
            )

        self.rgb_x_block = branch(5, 1)
        self.sfmap_block = branch(3, 3)
        self.softmax = nn.Softmax(dim=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, rgb, sparse_depth, sparse_mask):
        visual_input = torch.cat((rgb, sparse_depth, sparse_mask), dim=1)
        dense_depth = self.rgb_x_block(visual_input)
        maps = self.sfmap_block(torch.cat((sparse_depth, sparse_mask, dense_depth), dim=1))
        scale = torch.sigmoid(maps[:, 0:1]) * 2.0
        offset = F.hardtanh(maps[:, 1:2], min_val=-0.5, max_val=0.5)
        # Keep the source model's softmax axis for checkpoint compatibility.
        probability = self.softmax(maps[:, 2]).unsqueeze(1)
        output = (dense_depth * scale + offset) * probability + dense_depth * (1 - probability)
        return output, scale, offset, probability, dense_depth
