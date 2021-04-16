# pyre-strict
from typing import Generic, TypeVar
from typing_extensions import Literal as L
from pyre_extensions import Divide, Add, Multiply

import torch
from torch import Tensor


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


class TransformerNet(torch.nn.Module):
    def __init__(self) -> None:
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1: ConvLayer[L[3], L[32], L[9], L[1]] = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2: ConvLayer[L[32], L[64], L[3], L[2]] = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3: ConvLayer[L[64], L[128], L[3], L[2]] = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1: UpsampleConvLayer[L[128], L[64], L[3], L[1], L[2]] = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2: UpsampleConvLayer[L[64], L[32], L[3], L[1],  L[2]] = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3: ConvLayer[L[32], L[3], L[9], L[1]] = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X: Tensor[float, L[1], L[3], L[1080], L[1080]]) -> Tensor[float, L[1], L[3], L[1060], L[1060]]:
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


class ConvLayer(torch.nn.Module, Generic[InChannels, OutChannels, KernelSize, Stride]):

    def __init__(
            self,
            in_channels: InChannels,
            out_channels: OutChannels,
            kernel_size: KernelSize,
            stride: Stride,
    ) -> None:
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad: torch.nn.ReflectionPad2d[Divide[KernelSize, L[2]]] = torch.nn.ReflectionPad2d(reflection_padding)  # type: ignore
        self.conv2d: torch.nn.Conv2d[InChannels, OutChannels, KernelSize, Stride] = torch.nn.Conv2d(
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
        # (height - kernel_size + 2 * padding) / stride + 1, where padding=0 by default
        Add[Divide[Add[Height, Multiply[KernelSize, L[-1]]], Stride], L[1]],
        # (width - kernel_size + 2 * padding) / stride + 1, where padding=0 by default
        Add[Divide[Add[Width, Multiply[KernelSize, L[-1]]], Stride], L[1]],
    ]:
        return super()(x)

    def forward(self, x: Tensor[DType, Batch, InChannels, Height, Width]) -> Tensor[
        DType,
        Batch,
        OutChannels,
        # (height - kernel_size + 2 * padding) / stride + 1, where padding=0 by default
        Add[Divide[Add[Height, Multiply[KernelSize, L[-1]]], Stride], L[1]],
        # (width - kernel_size + 2 * padding) / stride + 1, where padding=0 by default
        Add[Divide[Add[Width, Multiply[KernelSize, L[-1]]], Stride], L[1]],
    ]:
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels: Channels) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1: ConvLayer[Channels, Channels, L[3], L[1]] = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2: ConvLayer[Channels, Channels, L[3], L[1]] = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

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

        # Batch x Channels x Height-4 x Width-4
        # reveal_type(out)
        # Batch x Channels x Height x Width
        # reveal_type(residual)

        # Not sure how this is not a runtime error.
        # REPL:
        # >>> torch.zeros(1, 2, 10, 10) + torch.zeros(1, 2, 6, 6)
        # RuntimeError: The size of tensor a (10) must match the size of
        # tensor b (6) at non-singleton dimension 3
        out = out + residual  # type: ignore
        return out  # type: ignore


class UpsampleConvLayer(torch.nn.Module, Generic[InChannels, OutChannels, KernelSize, Stride, Upscale]):
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
        self.reflection_pad: torch.nn.ReflectionPad2d[Divide[KernelSize, L[2]]] = torch.nn.ReflectionPad2d(reflection_padding)  # type: ignore
        self.conv2d: torch.nn.Conv2d[InChannels, OutChannels, KernelSize, Stride] = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

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
