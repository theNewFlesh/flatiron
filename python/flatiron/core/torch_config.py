from typing import Optional

import pydantic as pyd

OptBool = Optional[bool]
OptInt = Optional[int]
OptFloat = Optional[float]
OptStr = Optional[str]
# ------------------------------------------------------------------------------


class BaseConfig(pyd.BaseModel):
    name: str


# OPTIMIZER---------------------------------------------------------------------
class TorchBaseConfig(BaseConfig):
    lr: float = 0.01


class TMax(pyd.BaseModel):
    maximize: bool = False


class TFor(pyd.BaseModel):
    foreach: OptBool = None


class TDiff(pyd.BaseModel):
    differentiable: bool = False


class TEps(pyd.BaseModel):
    eps: float = 1e-06


class TCap(pyd.BaseModel):
    capturable: bool = False


class TDecay(pyd.BaseModel):
    weight_decay: float = 0


class TorchASGDConfig(TorchBaseConfig, TCap, TDecay, TDiff, TFor, TMax):
    alpha: float = 0.75
    lambd: float = 0.0001
    t0: float = 1000000.0


class TorchAdadeltaConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    rho: float = 0.9


class TorchAdafactorConfig(TorchBaseConfig, TDecay, TEps, TFor, TMax):
    beta2_decay: float = -0.8
    d: float = 1.0


class TorchAdagradConfig(TorchBaseConfig, TDecay, TDiff, TEps, TFor, TMax):
    fused: OptBool = None
    initial_accumulator_value: float = 0
    lr_decay: float = 0


class TorchAdamConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    amsgrad: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    fused: OptBool = None


class TorchAdamWConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    amsgrad: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    fused: OptBool = None


class TorchAdamaxConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    betas: tuple[float, float] = (0.9, 0.999)


class TorchLBFGSConfig(TorchBaseConfig):
    history_size: int = 100
    line_search_fn: OptStr = None
    max_eval: OptInt = None
    max_iter: int = 20
    tolerance_change: float = 1e-09
    tolerance_grad: float = 1e-07


class TorchNAdamConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    betas: tuple[float, float] = (0.9, 0.999)
    momentum_decay: float = 0.004


class TorchRAdamConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    betas: tuple[float, float] = (0.9, 0.999)


class TorchRMSpropConfig(TorchBaseConfig, TCap, TDecay, TDiff, TEps, TFor, TMax):
    alpha: float = 0.99
    centered: bool = False
    momentum: float = 0


class TorchRpropConfig(TorchBaseConfig, TCap, TDiff, TFor, TMax):
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-06, 50)


class TorchSGDConfig(TorchBaseConfig, TDecay, TDiff, TFor, TMax):
    dampening: float = 0
    fused: OptBool = None
    momentum: float = 0
    nesterov: bool = False


class TorchSparseAdamConfig(TorchBaseConfig, TEps, TMax):
    betas: tuple[float, float] = (0.9, 0.999)
