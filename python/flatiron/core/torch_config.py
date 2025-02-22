from flatiron.core.types import OptBool, OptInt, OptStr

import pydantic as pyd
# ------------------------------------------------------------------------------


# BASE--------------------------------------------------------------------------
class TorchBaseConfig(pyd.BaseModel):
    name: str


# OPTIMIZER HELPERS
class TorchOptBaseConfig(TorchBaseConfig):
    learning_rate: float = 0.01  # convert to lr


class TMax(pyd.BaseModel):
    maximize: bool = False


class TFor(pyd.BaseModel):
    foreach: OptBool = None


class TDiff(pyd.BaseModel):
    differentiable: bool = False


class TEps(pyd.BaseModel):
    epsilon: float = 1e-06  # convert to eps


class TCap(pyd.BaseModel):
    capturable: bool = False


class TDecay(pyd.BaseModel):
    weight_decay: float = 0


class TBeta(pyd.BaseModel):
    beta_1: float = 0.9
    beta_2: float = 0.999  # convert to betas: tuple[float, float]


class TGroup1(TCap, TDecay, TDiff, TEps, TFor, TMax):
    pass


# LOSS HELPERS
class TReduct(pyd.BaseModel):
    reduction: str = 'mean'


class TRed(pyd.BaseModel):
    reduce: OptBool = None


class TSize(pyd.BaseModel):
    size_average: OptBool = None


class TMarg(pyd.BaseModel):
    margin: float = 0.0


class TGroup2(TRed, TReduct, TSize):
    pass


class TGroup3(TMarg, TRed, TReduct, TSize):
    pass


# OPTIMIZER---------------------------------------------------------------------
class TorchOptASGDConfig(TorchOptBaseConfig, TCap, TDecay, TDiff, TFor, TMax):
    alpha: float = 0.75
    lambd: float = 0.0001
    t0: float = 1000000.0


class TorchOptAdadeltaConfig(TorchOptBaseConfig, TGroup1):
    rho: float = 0.9


class TorchOptAdafactorConfig(TorchOptBaseConfig, TDecay, TEps, TFor, TMax):
    beta2_decay: float = -0.8
    clipping_threshold: float = 1.0  # convert to d


class TorchOptAdagradConfig(TorchOptBaseConfig, TDecay, TDiff, TEps, TFor, TMax):
    fused: OptBool = None
    initial_accumulator_value: float = 0
    lr_decay: float = 0


class TorchOptAdamConfig(TorchOptBaseConfig, TGroup1, TBeta):
    amsgrad: bool = False
    fused: OptBool = None


class TorchOptAdamWConfig(TorchOptBaseConfig, TGroup1, TBeta):
    amsgrad: bool = False
    fused: OptBool = None


class TorchOptAdamaxConfig(TorchOptBaseConfig, TGroup1, TBeta):
    pass


class TorchOptLBFGSConfig(TorchOptBaseConfig):
    history_size: int = 100
    line_search_fn: OptStr = None
    max_eval: OptInt = None
    max_iter: int = 20
    tolerance_change: float = 1e-09
    tolerance_grad: float = 1e-07


class TorchOptNAdamConfig(TorchOptBaseConfig, TGroup1, TBeta):
    momentum_decay: float = 0.004


class TorchOptRAdamConfig(TorchOptBaseConfig, TGroup1, TBeta):
    pass


class TorchOptRMSpropConfig(TorchOptBaseConfig, TGroup1):
    alpha: float = 0.99
    centered: bool = False
    momentum: float = 0


class TorchOptRpropConfig(TorchOptBaseConfig, TCap, TDiff, TFor, TMax):
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-06, 50)


class TorchOptSGDConfig(TorchOptBaseConfig, TDecay, TDiff, TFor, TMax):
    dampening: float = 0
    fused: OptBool = None
    momentum: float = 0
    nesterov: bool = False


class TorchOptSparseAdamConfig(TorchOptBaseConfig, TEps, TMax, TBeta):
    pass


# LOSS--------------------------------------------------------------------------
class TorchBCELossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchBCEWithLogitsLossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchCTCLossConfig(TorchBaseConfig, TReduct):
    blank: int = 0
    zero_infinity: bool = False


class TorchCosineEmbeddingLossConfig(TorchBaseConfig, TGroup3):
    pass


class TorchCrossEntropyLossConfig(TorchBaseConfig, TGroup2):
    ignore_index: int = -100
    label_smoothing: float = 0.0


class TorchGaussianNLLLossConfig(TorchBaseConfig, TEps, TReduct):
    full: bool = False


class TorchHingeEmbeddingLossConfig(TorchBaseConfig, TGroup3):
    pass


class TorchHuberLossConfig(TorchBaseConfig, TReduct):
    delta: float = 1.0


class TorchKLDivLossConfig(TorchBaseConfig, TGroup2):
    log_target: bool = False


class TorchL1LossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchMSELossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchMarginRankingLossConfig(TorchBaseConfig, TGroup3):
    pass


class TorchMultiLabelMarginLossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchMultiLabelSoftMarginLossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchMultiMarginLossConfig(TorchBaseConfig, TGroup3):
    exponent: int = 1  # convert to p


class TorchNLLLossConfig(TorchBaseConfig, TGroup2):
    ignore_index: int = -100


class TorchPairwiseDistanceConfig(TorchBaseConfig, TEps):
    keepdim: bool = False
    norm_degree: float = 2.0  # convert to p


class TorchPoissonNLLLossConfig(TorchBaseConfig, TEps, TGroup2):
    full: bool = False
    log_input: bool = True


class TorchSmoothL1LossConfig(TorchBaseConfig, TGroup2):
    beta: float = 1.0


class TorchSoftMarginLossConfig(TorchBaseConfig, TGroup2):
    pass


class TorchTripletMarginLossConfig(TorchBaseConfig, TEps, TGroup3):
    norm_degree: float = 2.0  # convert to p
    swap: bool = False


class TorchTripletMarginWithDistanceLossConfig(TorchBaseConfig, TMarg, TReduct):
    swap: bool = False
