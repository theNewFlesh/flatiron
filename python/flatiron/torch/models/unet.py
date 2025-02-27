from typing import Callable  # noqa F401
from typing_extensions import Annotated

import pydantic as pyd

import torch
import torch.nn as nn

import flatiron.core.validators as vd
import flatiron.core.pipeline as ficp
# ------------------------------------------------------------------------------


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, filters=16, dtype=torch.float16):
        super().__init__()
        kwargs = dict(
            out_channels=filters, kernel_size=(3, 3),
            stride=(1, 1), padding=1, padding_mode='reflect', dtype=dtype
        )
        self.conv_1 = nn.Conv2d(in_channels=in_channels, **kwargs)
        self.act_1 = nn.ReLU()
        self.batch_1 = nn.BatchNorm2d(filters, dtype=dtype)
        self.act_1 = nn.Sigmoid()
        self.conv_2 = nn.Conv2d(in_channels=filters, **kwargs)
        self.act_2 = nn.ReLU()
        self.batch_2 = nn.BatchNorm2d(filters, dtype=dtype)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.batch_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.batch_2(x)
        return x


class AtttentionGate2DBlock(nn.Module):
    def __init__(self, in_channels, filters=16, dtype=torch.float16):
        super().__init__()
        kwargs = dict(
            kernel_size=(3, 3),
            stride=(1, 1), padding=1, padding_mode='reflect', dtype=dtype
        )
        self.conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=filters, **kwargs)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, **kwargs)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=filters, out_channels=1, **kwargs)
        self.act_1 = nn.Sigmoid()

    def forward(self, skip_connection, query):
        skip = self.conv_0(skip_connection)
        query = self.conv_1(query)

        gate = torch.add(skip, query)
        gate = self.act_1(gate)
        gate = self.conv_2(gate)
        gate = self.act_2(gate)
        gate = torch.multiply(skip, gate)

        x = torch.concatenate([gate, query])
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, attention=False, dtype=torch.float16):
        super().__init__()
        self._attention = attention
        kwargs = dict(dtype=dtype)
        pool_kwargs = dict(kernel_size=(2, 2), stride=(2, 2))
        trans_kwargs = dict(kernel_size=(2, 2), stride=(2, 2), dtype=dtype)

        self.encode_block_00 = Conv2DBlock(in_channels=in_channels, filters=16, **kwargs)
        self.downsample_00 = nn.MaxPool2d(**pool_kwargs)
        self.encode_block_01 = Conv2DBlock(in_channels=16, filters=32, **kwargs)
        self.downsample_01 = nn.MaxPool2d(**pool_kwargs)

        self.middle_block = Conv2DBlock(in_channels=32, filters=64, **kwargs)

        self.upsample_00 = nn.ConvTranspose2d(in_channels=64, out_channels=32, **trans_kwargs)
        self.decode_block_00 = Conv2DBlock(in_channels=64, filters=32, **kwargs)
        self.upsample_01 = nn.ConvTranspose2d(in_channels=32, out_channels=16, **trans_kwargs)
        self.decode_block_01 = Conv2DBlock(in_channels=32, filters=out_channels, **kwargs)

    def forward(self, x):
        x0 = self.encode_block_00(x)
        x = self.downsample_00(x0)
        x1 = self.encode_block_01(x)
        x = self.downsample_01(x1)
        x = self.middle_block(x)

        x = self.upsample_00(x)
        x = torch.concatenate([x, x1], axis=1)
        x = self.decode_block_00(x)
        x = self.upsample_01(x)
        x = torch.concatenate([x, x0], axis=1)
        x = self.decode_block_01(x)
        return x


def get_unet_model(in_channels, out_channels=3, dtype='float16'):
    return UNet(in_channels, out_channels, dtype)


# CONFIG------------------------------------------------------------------------
class UNetConfig(pyd.BaseModel):
    '''
    Configuration for UNet model.

    Attributes:
        input_width (int): Input width.
        input_height (int): Input height.
        input_channels (int): Input channels.
        classes (int, optional): Number of output classes. Default: 1.
        filters (int, optional): Number of filters for initial con 2d block.
            Default: 16.
        layers (int, optional): Total number of layers. Default: 9.
        activation (KerasTensor, optional): Activation function to be used.
            Default: relu.
        batch_norm (KerasTensor, optional): Use batch normalization.
            Default: True.
        output_activation (KerasTensor, optional): Output activation function.
            Default: sigmoid.
        kernel_initializer (KerasTensor, optional): Default: he_normal.
        attention_gates (KerasTensor, optional): Use attention gates.
            Default: False.
        attention_activation_1 (str, optional): First activation.
            Default: 'relu'
        attention_activation_2 (str, optional): Second activation.
            Default: 'sigmoid'
        attention_kernel_size (int, optional): Kernel_size. Default: 1
        attention_strides (int, optional): Strides. Default: 1
        attention_padding (str, optional): Padding. Default: 'same'
        attention_kernel_initializer (str, optional): Kernel initializer.
            Default: 'he_normal'
    '''
    input_width: Annotated[int, pyd.Field(ge=1)]
    input_height: Annotated[int, pyd.Field(ge=1)]
    input_channels: Annotated[int, pyd.Field(ge=1)]
    classes: Annotated[int, pyd.Field(ge=1)] = 1
    filters: Annotated[int, pyd.Field(ge=1)] = 16
    layers: Annotated[int, pyd.Field(ge=3), pyd.AfterValidator(vd.is_odd)] = 9
    activation: str = 'relu'
    batch_norm: bool = True
    output_activation: str = 'sigmoid'
    kernel_initializer: str = 'he_normal'
    attention_gates: bool = False
    attention_activation_1: str = 'relu'
    attention_activation_2: str = 'sigmoid'
    attention_kernel_size: Annotated[int, pyd.Field(ge=1)] = 1
    attention_strides: Annotated[int, pyd.Field(ge=1)] = 1
    attention_padding: Annotated[str, pyd.AfterValidator(vd.is_padding)] = 'same'
    attention_kernel_initializer: str = 'he_normal'
    dtype: str = 'float16'
    data_format: str = 'channels_last'


# PIPELINE----------------------------------------------------------------------
class UNetPipeline(ficp.PipelineBase):
    def model_config(self):
        # type: () -> type[UNetConfig]
        return UNetConfig

    def model_func(self):
        # type: () -> Callable[..., nn.Module]
        return get_unet_model
