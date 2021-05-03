# pyre-strict
from typing import Generic, TypeVar
from typing_extensions import Literal as L
from pyre_extensions import Divide, Add, Multiply

import torch
from torch import Tensor
from torch.nn import Conv2d, InstanceNorm2d, Module, ReLU, ReflectionPad2d


DType = TypeVar('DType')
InChannels = TypeVar('InChannels', bound=int)
OutChannels = TypeVar('OutChannels', bound=int)
KernelSize = TypeVar('KernelSize', bound=int)
Stride = TypeVar('Stride', bound=int)
Batch = TypeVar('Batch', bound=int)
Height = TypeVar('Height', bound=int)
Width = TypeVar('Width', bound=int)
Channels = TypeVar('Channels', bound=int)
ReflectionPadding = TypeVar('ReflectionPadding', bound=int)
Upscale = TypeVar('Upscale', bound=int)


class TransformerNet(Module):
    def __init__(self) -> None:
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1: ConvLayer[L[3], L[32], L[9], L[1]] = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1: InstanceNorm2d[L[32]] = InstanceNorm2d(32, affine=True)
        self.conv2: ConvLayer[L[32], L[64], L[3], L[2]] = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2: InstanceNorm2d[L[64]] = InstanceNorm2d(64, affine=True)
        self.conv3: ConvLayer[L[64], L[128], L[3], L[2]] = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3: InstanceNorm2d[L[128]] = InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1: ResidualBlock[L[128]] = ResidualBlock(128)
        self.res2: ResidualBlock[L[128]] = ResidualBlock(128)
        self.res3: ResidualBlock[L[128]] = ResidualBlock(128)
        self.res4: ResidualBlock[L[128]] = ResidualBlock(128)
        self.res5: ResidualBlock[L[128]] = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1: UpsampleConvLayer[L[128], L[64], L[3], L[1], L[2]] = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4: InstanceNorm2d[L[64]] = InstanceNorm2d(64, affine=True)
        self.deconv2: UpsampleConvLayer[L[64], L[32], L[3], L[1],  L[2]] = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5: InstanceNorm2d[L[32]] = InstanceNorm2d(32, affine=True)
        self.deconv3: ConvLayer[L[32], L[3], L[9], L[1]] = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = ReLU()

    def forward(self, X: Tensor[float, L[1], L[3], L[1080], L[1080]]) -> Tensor[float, L[1], L[3], L[1080], L[1080]]:
        y1 = self.relu(self.in1(self.conv1(X)))
        y2 = self.relu(self.in2(self.conv2(y1)))
        y3 = self.relu(self.in3(self.conv3(y2)))
        y4 = self.res1(y3)
        y5 = self.res2(y4)
        y6 = self.res3(y5)
        y7 = self.res4(y6)
        y8 = self.res5(y7)
        y9 = self.relu(self.in4(self.deconv1(y8)))
        y10 = self.relu(self.in5(self.deconv2(y9)))
        y11 = self.deconv3(y10)
        return y11


class ConvLayer(Module, Generic[InChannels, OutChannels, KernelSize, Stride]):

    def __init__(
            self,
            in_channels: InChannels,
            out_channels: OutChannels,
            kernel_size: KernelSize,
            stride: Stride,
    ) -> None:
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad: ReflectionPad2d[Divide[KernelSize, L[2]]] = ReflectionPad2d(reflection_padding)  # type: ignore
        self.conv2d: Conv2d[InChannels, OutChannels, KernelSize, Stride] = Conv2d(
            in_channels, out_channels, kernel_size, stride
        )

    # Note that we have to specify the signature of `__call__` directly -
    # unfortunately, Pyre can't infer the signature of `__call__` based on
    # the signature of `forward`.
    def __call__(
            self,
            x: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Add[Divide[Add[Add[Height, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
        Add[Divide[Add[Add[Width, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
    ]:
        return super()(x)

    def forward(self, x: Tensor[DType, Batch, InChannels, Height, Width]) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Add[Divide[Add[Add[Height, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
        Add[Divide[Add[Add[Width, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
    ]:
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(Module, Generic[Channels]):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels: Channels) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1: ConvLayer[Channels, Channels, L[3], L[1]] = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1: InstanceNorm2d[Channels] = InstanceNorm2d(channels, affine=True)
        self.conv2: ConvLayer[Channels, Channels, L[3], L[1]] = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2: InstanceNorm2d[Channels] = InstanceNorm2d(channels, affine=True)
        self.relu = ReLU()

    # Note that, as with `ConvLayer`, we have to specify the signature
    # here, not `forward`.
    def __call__(
            self,
            x: Tensor[DType, Batch, Channels, Height, Width]
    ) -> Tensor[DType, Batch, Channels, Height, Width]:
        return super()(x)

    def forward(self, x: Tensor[DType, Batch, Channels, Height, Width]) -> Tensor[DType, Batch, Channels, Height, Width]:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        # pyre-ignore[58]: Bug in Pyre where N + 1 - 1 is not treated as compatible with N.
        out = out + residual
        # pyre-ignore[7]: Bug in Pyre where N + 1 - 1 is not treated as compatible with N.
        return out


class UpsampleConvLayer(Module, Generic[InChannels, OutChannels, KernelSize, Stride, Upscale]):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
            self,
            in_channels: InChannels,
            out_channels: OutChannels,
            kernel_size: KernelSize,
            stride: Stride,
            upsample: Upscale,
    ) -> None:
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad: ReflectionPad2d[Divide[KernelSize, L[2]]] = ReflectionPad2d(reflection_padding)  # type: ignore
        self.conv2d: Conv2d[InChannels, OutChannels, KernelSize, Stride] = Conv2d(in_channels, out_channels, kernel_size, stride)

    # Note that, as with `ConvLayer` and `ResidualBlock`,
    # we need to specify the signature of `forward` here instead.
    def __call__(
            self,
            input: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Multiply[Height, Upscale],
        Multiply[Width, Upscale]
    ]:
        return super()(input)

    def forward(self, x: Tensor[DType, Batch, InChannels, Height, Width]) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Multiply[Height, Upscale],
        Multiply[Width, Upscale]
    ]:
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode="nearest", scale_factor=self.upsample
            )
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
