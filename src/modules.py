from torch import Tensor
import torch.nn as nn
from torch.nn import Module, Conv2d, Sequential, UpsamplingNearest2d
from .custom_blocks import BottleNeck, ResNeXtBottleneck, CNBlock, NormLayer
import torch

class Encoder(nn.Module):
    def __init__(self, in_chans=5, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], dp_rate=0.0, norm_type='CNX'):
        super(Encoder, self).__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        # print(all_dims)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, all_dims[0], kernel_size=3, padding=1)
        self.downsample_layers.append(stem)
        for i in range(5):
            downsample_layer = nn.Sequential(
                NormLayer(all_dims[i], norm_type),
                nn.Conv2d(all_dims[i], all_dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[CNBlock(dims[i], norm_type, dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        outputs = []
        for i in range(6):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, out_chans=1, dims=[96, 192, 384, 768], norm_type='CNX'):
        super(Decoder, self).__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.upsample_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for i in range(5):
            upsample_layer = nn.ConvTranspose2d(all_dims[i + 1], all_dims[i], kernel_size=2, stride=2)
            fusion_layer = nn.Conv2d(2 * all_dims[i], all_dims[i], kernel_size=1)
            self.upsample_layers.append(upsample_layer)
            self.fusion_layers.append(fusion_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        for i in range(3):
            stage = CNBlock(dims[i], norm_type, 0.)
            self.stages.append(stage)
        self.head = nn.Conv2d(all_dims[0], out_chans, kernel_size=3, padding=1)

    def forward(self, ins):
        x = ins[-1]
        for i in range(4, -1, -1):
            x = self.upsample_layers[i](x)
            x = torch.cat([ins[i], x], dim=1)
            x = self.fusion_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x
  
class StackedBottleNeck(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            rezero: bool,
    ) -> None:
        super(StackedBottleNeck, self).__init__()

        self.block = Sequential(
            BottleNeck(in_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.block(x)
        return x1


class StackedResNeXtBlock(Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            rezero: bool,

    ):
        super(StackedResNeXtBlock, self).__init__()
        self.block = Sequential(
            ResNeXtBottleneck(in_channel, out_channel, rezero),
            ResNeXtBottleneck(out_channel, out_channel, rezero),
            ResNeXtBottleneck(out_channel, out_channel, rezero),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.block(x)
        return x1


basic_block = StackedBottleNeck


class FirstModule(Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            rezero: bool,
    ):
        super(FirstModule, self).__init__()
        # self.skip = basic_block(in_channel, out_channel, rezero, lrelu)
        self.block = Sequential(
            Conv2d(in_channel, out_channel, 3, 2, 1),
            basic_block(out_channel, out_channel, rezero),
            UpsamplingNearest2d(scale_factor=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = torch.cat((x, self.block(x)), dim=1)
        return x1


class UNetModule(Module):
    def __init__(
            self,
            mid_module: Module,
            in_channel: int,
            mid_channel: int,
            rezero: bool,
    ):
        super(UNetModule, self).__init__()
        # self.skip = basic_block(in_channel, in_channel, rezero, lrelu)
        self.block = Sequential(
            Conv2d(in_channel, mid_channel, 3, 2, 1),
            basic_block(mid_channel, mid_channel, rezero),
            mid_module,
            basic_block(2 * mid_channel, in_channel, rezero),
            UpsamplingNearest2d(scale_factor=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = torch.cat((x, self.block(x)), dim=1)
        return x1
