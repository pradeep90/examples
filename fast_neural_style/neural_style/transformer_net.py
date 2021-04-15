from typing import Generic, overload, TypeVar
from typing_extensions import Literal, TYPE_CHECKING
from pyre_extensions import Divide, Add, Multiply

import torch
from torch import Tensor


L1 = Literal[1]
L2 = Literal[2]
L3 = Literal[3]
L9 = Literal[9]
L32 = Literal[32]
L64 = Literal[64]
L128 = Literal[128]
L270 = Literal[270]
L540 = Literal[540]
L1080 = Literal[1080]

DType = TypeVar('DType')
InChannels = TypeVar('InChannels', bound=int)
OutChannels = TypeVar('OutChannels', bound=int)
KernelSize = TypeVar('KernelSize', bound=int)
Stride = TypeVar('Stride', bound=int)
Batch = TypeVar('Batch', bound=int)
Height = TypeVar('Height', bound=int)
Width = TypeVar('Width', bound=int)
Channels = TypeVar('Channels', bound=int)
ReflectionPadding = TypeVar('ReflectionPadding', bount=int)
Upscale = TypeVar('Upscale', bound=int)


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1: ConvLayer[L3, L32, L9, L1] = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2: ConvLayer[L32, L64, L3, L2] = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3: ConvLayer[L64, L128, L3, L2] = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1: UpsampleConvLayer[L128, L64, L3, L1, L2] = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2: UpsampleConvLayer[L64, L32, L3, L1,  L2] = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3: ConvLayer[L32, L3, L9, L1] = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X: Tensor[float, L1, L3, L1080, L1080]):
        y1: Tensor[float, L1, L32, L1080, L1080] = self.relu(self.in1(self.conv1(X)))
        y2: Tensor[float, L1, L64, L540, L540] = self.relu(self.in2(self.conv2(y1)))
        y3: Tensor[float, L1, L128, L270, L270] = self.relu(self.in3(self.conv3(y2)))
        y4: Tensor[float, L1, L128, L270, L270] = self.res1(y3)
        y5: Tensor[float, L1, L128, L270, L270] = self.res2(y4)
        y6: Tensor[float, L1, L128, L270, L270] = self.res3(y5)
        y7: Tensor[float, L1, L128, L270, L270] = self.res4(y6)
        y8: Tensor[float, L1, L128, L270, L270] = self.res5(y7)
        y9: Tensor[float, L1, L64, L540, L540] = self.relu(self.in4(self.deconv1(y8)))
        y10: Tensor[float, L1, L32, L1080, L1080] = self.relu(self.in5(self.deconv2(y9)))
        y11: Tensor[float, L1, L3, L1080, L1080] = self.deconv3(y10)
        return y11


class ConvLayer(torch.nn.Module, Generic[InChannels, OutChannels, KernelSize, Stride]):

    def __init__(
            self,
            in_channels: InChannels,
            out_channels: OutChannels,
            kernel_size: KernelSize,
            stride: Stride,
    ):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def __call__(
            self,
            x: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        # (height - kernel_size + 2 * padding) / stride + 1
        Add[Divide[Add[Add[Height, Multiply[KernelSize, Literal[-1]]], Multiply[Divide[KernelSize, Literal[2]], Literal[2]]], Stride], Literal[1]],
        Add[Divide[Add[Add[Width, Multiply[KernelSize, Literal[-1]]], Multiply[Divide[KernelSize, Literal[2]], Literal[2]]], Stride], Literal[1]],

    ]: ...

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels: Channels):
        super(ResidualBlock, self).__init__()
        self.conv1: ConvLayer[Channels, Channels, L3, L1] = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2: ConvLayer[Channels, Channels, L3, L1] = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    # TODO(mrahtz): Figure out why Pyre can't figure this out automatically.
    def __call__(self, x: Tensor[DType, Batch, Channels, Height, Width]) -> Tensor[DType, Batch, Channels, Height, Width]: ...

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


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
    ):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def __call__(
            self,
            input: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Multiply[Height, Upscale],
        Multiply[Width, Upscale]
    ]: ...

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode="nearest", scale_factor=self.upsample
            )
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
