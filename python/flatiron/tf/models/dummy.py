from tensorflow import keras  # noqa F401
from keras import layers as tfl
from keras import models as tfmodels
import pydantic as pyd

import flatiron.core.pipeline as ficp
# ------------------------------------------------------------------------------


def get_dummy_model(shape):
    input_ = tfl.Input(shape, name='input')
    output = tfl.Conv2D(1, (1, 1), activation='relu', name='output')(input_)
    model = tfmodels.Model(inputs=[input_], outputs=[output])
    return model


class DummyConfig(pyd.BaseModel):
    shape: tuple[int]


class DummyPipeline(ficp.PipelineBase):
    def model_config(self):
        return DummyConfig

    def model_func(self):
        return get_dummy_model
