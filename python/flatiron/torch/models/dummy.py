import pydantic as pyd
import torch
import torch.nn as nn

import flatiron.core.pipeline as ficp
# ------------------------------------------------------------------------------


class DummyModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels,
                kernel_size=(3, 3), dtype=torch.float16, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer_stack(x)


def get_dummy_model(input_channels=3, output_channels=1):
    return DummyModel(
        input_channels=input_channels,
        output_channels=output_channels,
    )


class DummyConfig(pyd.BaseModel):
    input_channels: int
    output_channels: int


class DummyPipeline(ficp.PipelineBase):
    def model_config(self):
        return DummyConfig

    def model_func(self):
        return get_dummy_model
