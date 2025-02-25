from tensorflow import keras  # noqa F401
from keras import layers as tfl
from keras import models as tfmodels
import pydantic as pyd

import flatiron.core.pipeline as ficp
# ------------------------------------------------------------------------------


def get_dummy_model(shape, activation='relu'):
    input_ = tfl.Input(shape, name='input')
    output = tfl.Conv2D(1, (1, 1), activation=activation, name='output')(input_)
    model = tfmodels.Model(inputs=[input_], outputs=[output])
    return model


class DummyConfig(pyd.BaseModel):
    shape: list[int]
    activation: str = 'relu'


class DummyPipeline(ficp.PipelineBase):
    def model_config(self):
        return DummyConfig

    def model_func(self):
        return get_dummy_model
