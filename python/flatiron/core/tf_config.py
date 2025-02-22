from typing import Optional

import pydantic as pyd

OptBool = Optional[bool]
OptInt = Optional[int]
OptFloat = Optional[float]
# ------------------------------------------------------------------------------


class BaseConfig(pyd.BaseModel):
    name: str


# OPTIMIZER---------------------------------------------------------------------
class TFOptBaseConfig(BaseConfig):
    clipnorm: OptFloat = None
    clipvalue: OptFloat = None
    ema_momentum: float = 0.99
    ema_overwrite_frequency: OptInt = None
    global_clipnorm: OptFloat = None
    gradient_accumulation_steps: OptInt = None
    learning_rate: float = 0.001
    loss_scale_factor: OptFloat = None
    use_ema: bool = False
    weight_decay: OptFloat = None


class TFEpsilon(pyd.BaseModel):
    epsilon: float = 1e-07


class TFBeta(pyd.BaseModel):
    beta_1: float = 0.9
    beta_2: float = 0.99
# ------------------------------------------------------------------------------


class TFOptAdafactorConfig(TFOptBaseConfig):
    beta_2_decay: float = -0.8
    clip_threshold: float = 1.0
    epsilon_1: float = 1e-30
    epsilon_2: float = 0.001
    relative_step: bool = True


class TFOptFtrlConfig(TFOptBaseConfig):
    beta: float = 0.0
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    l2_shrinkage_regularization_strength: float = 0.0
    learning_rate_power: float = -0.5


class TFOptLionConfig(TFOptBaseConfig, TFBeta):
    pass


class TFOptSGDConfig(TFOptBaseConfig):
    momentum: float = 0.0
    nesterov: bool = False


class TFOptAdadeltaConfig(TFOptBaseConfig, TFEpsilon):
    rho: float = 0.95


class TFOptAdagradConfig(TFOptBaseConfig, TFEpsilon):
    initial_accumulator_value: float = 0.1


class TFOptAdamConfig(TFOptBaseConfig, TFBeta, TFEpsilon):
    amsgrad: bool = False


class TFOptAdamWConfig(TFOptBaseConfig, TFBeta, TFEpsilon):
    amsgrad: bool = False


class TFOptAdamaxConfig(TFOptBaseConfig, TFBeta, TFEpsilon):
    pass


class TFOptLambConfig(TFOptBaseConfig, TFBeta, TFEpsilon):
    pass


class TFOptNadamConfig(TFOptBaseConfig, TFBeta, TFEpsilon):
    pass


class TFOptRMSpropConfig(TFOptBaseConfig, TFEpsilon):
    centered: bool = False
    momentum: float = 0.0
    rho: float = 0.9


# LOSS--------------------------------------------------------------------------
class TFLossBaseConfig(BaseConfig):
    dtype: Optional[str] = None
    reduction: str = 'sum_over_batch_size'


class TFAxis(pyd.BaseModel):
    axis: int = -1


class TFLogits(pyd.BaseModel):
    from_logits: bool = False
# ------------------------------------------------------------------------------


class TFLossBinaryCrossentropyConfig(TFLossBaseConfig, TFAxis, TFLogits):
    label_smoothing: float = 0.0


class TFLossBinaryFocalCrossentropyConfig(TFLossBaseConfig, TFAxis, TFLogits):
    alpha: float = 0.25
    apply_class_balancing: bool = False
    gamma: float = 2.0
    label_smoothing: float = 0.0


class TFLossCategoricalCrossentropyConfig(TFLossBaseConfig, TFAxis, TFLogits):
    label_smoothing: float = 0.0


class TFLossCategoricalFocalCrossentropyConfig(TFLossBaseConfig, TFAxis, TFLogits):
    alpha: float = 0.25
    gamma: float = 2.0
    label_smoothing: float = 0.0


class TFLossCircleConfig(TFLossBaseConfig):
    gamma: float = 80.0
    margin: float = 0.4
    remove_diagonal: bool = True


class TFLossCosineSimilarityConfig(TFLossBaseConfig, TFAxis):
    pass


class TFLossDiceConfig(TFLossBaseConfig, TFAxis):
    pass


class TFLossHuberConfig(TFLossBaseConfig):
    delta: float = 1.0


class TFLossSparseCategoricalCrossentropyConfig(TFLossBaseConfig, TFLogits):
    ignore_class: Optional[int] = None


class TFLossTverskyConfig(TFLossBaseConfig, TFAxis):
    alpha: float = 0.5
    beta: float = 0.5
