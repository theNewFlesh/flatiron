from typing import Optional, Union
from flatiron.core.types import OptBool, OptFloat, OptInt, OptStr

import pydantic as pyd
# ------------------------------------------------------------------------------


# BASE--------------------------------------------------------------------------
class TorchBaseConfig(pyd.BaseModel):
    name: str


# OPTIMIZER-HELPERS-------------------------------------------------------------
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


# LOSS-HELPERS------------------------------------------------------------------
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


# METRIC-HELPERS----------------------------------------------------------------
class TInd(pyd.BaseModel):
    ignore_index: OptInt = None


class TNan(pyd.BaseModel):
    nan_strategy: Union[str, float] = 'warn'  # has multiple signatures


class TAct(pyd.BaseModel):
    empty_target_action: str = 'neg'  # has multiple signatures


class TOut(pyd.BaseModel):
    num_outputs: int = 1


class TMReduct(pyd.BaseModel):
    reduction: OptStr = 'elementwise_mean'  # has multiple signatures


class TTopK(pyd.BaseModel):
    top_k: OptInt = None


class TCls(pyd.BaseModel):
    num_classes: OptInt = None  # has multiple signatures


class TDate(pyd.BaseModel):
    data_range: Optional[Union[float, tuple[float, float]]] = None


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


# METRICS-----------------------------------------------------------------------
class TorchMetBLEUScoreConfig(TorchBaseConfig):
    n_gram: int = 4
    smooth: bool = False
    weights: Optional[list[float]] = None


class TorchMetCHRFScoreConfig(TorchBaseConfig):
    beta: float = 2.0
    lowercase: bool = False
    n_char_order: int = 6
    n_word_order: int = 2
    return_sentence_level_score: bool = False
    whitespace: bool = False


class TorchMetCatMetricConfig(TorchBaseConfig, TNan):
    pass


class TorchMetConcordanceCorrCoefConfig(TorchBaseConfig, TOut):
    pass


class TorchMetCosineSimilarityConfig(TorchBaseConfig):
    reduction: str = 'sum'


class TorchMetCramersVConfig(TorchBaseConfig, TCls, TNan):
    bias_correction: bool = True
    nan_replace_value: OptFloat = 0.0


class TorchMetCriticalSuccessIndexConfig(TorchBaseConfig):
    keep_sequence_dim: OptInt = None
    threshold: float


class TorchMetDiceConfig(TorchBaseConfig, TCls, TInd, TTopK):
    average: OptStr = 'micro'
    mdmc_average: OptStr = 'global'
    multiclass: OptBool = None
    threshold: float = 0.5
    zero_division: int = 0


class TorchMetErrorRelativeGlobalDimensionlessSynthesisConfig(TorchBaseConfig, TMReduct):
    ratio: float = 4


class TorchMetExplainedVarianceConfig(TorchBaseConfig):
    multioutput: str = 'uniform_average'


class TorchMetExtendedEditDistanceConfig(TorchBaseConfig):
    alpha: float = 2.0
    deletion: float = 0.2
    insertion: float = 1.0
    language: str = 'en'
    return_sentence_level_score: bool = False
    rho: float = 0.3


class TorchMetFleissKappaConfig(TorchBaseConfig):
    mode: str = 'counts'


class TorchMetKLDivergenceConfig(TorchBaseConfig):
    log_prob: bool = False
    reduction: str = 'mean'


class TorchMetKendallRankCorrCoefConfig(TorchBaseConfig, TOut):
    alternative: OptStr = 'two-sided'
    t_test: bool = False
    variant: str = 'b'


class TorchMetLogCoshErrorConfig(TorchBaseConfig, TOut):
    pass


class TorchMetMaxMetricConfig(TorchBaseConfig, TNan):
    pass


class TorchMetMeanAbsoluteErrorConfig(TorchBaseConfig, TOut):
    pass


class TorchMetMeanMetricConfig(TorchBaseConfig, TNan):
    pass


class TorchMetMeanSquaredErrorConfig(TorchBaseConfig, TOut):
    squared: bool = True


class TorchMetMinMetricConfig(TorchBaseConfig, TNan):
    pass


class TorchMetMinkowskiDistanceConfig(TorchBaseConfig):
    p: float


class TorchMetModifiedPanopticQualityConfig(TorchBaseConfig):
    allow_unknown_preds_category: bool = False
    stuffs: list[int]
    things: list[int]


class TorchMetMultiScaleStructuralSimilarityIndexMeasureConfig(TorchBaseConfig, TMReduct, TDate):
    betas: tuple = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    gaussian_kernel: bool = True
    k1: float = 0.01
    k2: float = 0.03
    kernel_size: Union[int, list[int]] = 11
    normalize: str = 'relu'
    sigma: Union[float, list[float]] = 1.5


class TorchMetNormalizedRootMeanSquaredErrorConfig(TorchBaseConfig, TOut):
    normalization: str = 'mean'


class TorchMetPanopticQualityConfig(TorchBaseConfig):
    allow_unknown_preds_category: bool = False
    stuffs: list[int]
    things: list[int]


class TorchMetPeakSignalNoiseRatioConfig(TorchBaseConfig, TMReduct, TDate):
    base: float = 10.0
    dim: Optional[Union[int, tuple[int, ...]]] = None


class TorchMetPearsonCorrCoefConfig(TorchBaseConfig, TOut):
    pass


class TorchMetPearsonsContingencyCoefficientConfig(TorchBaseConfig):
    nan_replace_value: OptFloat = 0.0
    nan_strategy: str = 'replace'
    num_classes: int


class TorchMetPermutationInvariantTrainingConfig(TorchBaseConfig):
    eval_func: str = 'max'
    mode: str = 'speaker-wise'


class TorchMetPerplexityConfig(TorchBaseConfig, TInd):
    pass


class TorchMetR2ScoreConfig(TorchBaseConfig):
    adjusted: int = 0
    multioutput: str = 'uniform_average'


class TorchMetRelativeAverageSpectralErrorConfig(TorchBaseConfig):
    window_size: int = 8


class TorchMetRelativeSquaredErrorConfig(TorchBaseConfig, TOut):
    squared: bool = True


class TorchMetRetrievalFallOutConfig(TorchBaseConfig, TInd, TTopK):
    empty_target_action: str = 'pos'


class TorchMetRetrievalHitRateConfig(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetRetrievalMAPConfig(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetRetrievalMRRConfig(TorchBaseConfig, TAct, TInd):
    pass


class TorchMetRetrievalNormalizedDCGConfig(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetRetrievalPrecisionConfig(TorchBaseConfig, TAct, TInd, TTopK):
    adaptive_k: bool = False


class TorchMetRetrievalPrecisionRecallCurveConfig(TorchBaseConfig, TInd):
    adaptive_k: bool = False
    max_k: OptInt = None


class TorchMetRetrievalRPrecisionConfig(TorchBaseConfig, TAct, TInd):
    pass


class TorchMetRetrievalRecallConfig(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetRetrievalRecallAtFixedPrecisionConfig(TorchBaseConfig, TAct, TInd):
    adaptive_k: bool = False
    max_k: OptInt = None
    min_precision: float = 0.0


class TorchMetRootMeanSquaredErrorUsingSlidingWindowConfig(TorchBaseConfig):
    window_size: int = 8


class TorchMetRunningMeanConfig(TorchBaseConfig, TNan):
    window: int = 5


class TorchMetRunningSumConfig(TorchBaseConfig, TNan):
    window: int = 5


class TorchMetSacreBLEUScoreConfig(TorchBaseConfig):
    lowercase: bool = False
    n_gram: int = 4
    smooth: bool = False
    tokenize: str = '13a'
    weights: Optional[list[float]] = None


class TorchMetScaleInvariantSignalDistortionRatioConfig(TorchBaseConfig):
    zero_mean: bool = False


class TorchMetSignalDistortionRatioConfig(TorchBaseConfig):
    filter_length: int = 512
    load_diag: OptFloat = None
    use_cg_iter: OptInt = None
    zero_mean: bool = False


class TorchMetSignalNoiseRatioConfig(TorchBaseConfig):
    zero_mean: bool = False


class TorchMetSpearmanCorrCoefConfig(TorchBaseConfig, TOut):
    pass


class TorchMetSpectralAngleMapperConfig(TorchBaseConfig, TMReduct):
    pass


class TorchMetSpectralDistortionIndexConfig(TorchBaseConfig, TMReduct):
    p: int = 1


class TorchMetStructuralSimilarityIndexMeasureConfig(TorchBaseConfig, TMReduct):
    data_range: Optional[Union[float, tuple[float, float]]] = None
    gaussian_kernel: bool = True
    k1: float = 0.01
    k2: float = 0.03
    kernel_size: Union[int, list[int]] = 11
    return_contrast_sensitivity: bool = False
    return_full_image: bool = False
    sigma: Union[float, list[float]] = 1.5


class TorchMetSumMetricConfig(TorchBaseConfig, TNan):
    pass


class TorchMetTheilsUConfig(TorchBaseConfig):
    nan_replace_value: OptFloat = 0.0
    nan_strategy: str = 'replace'
    num_classes: int


class TorchMetTotalVariationConfig(TorchBaseConfig):
    reduction: str = 'sum'


class TorchMetTranslationEditRateConfig(TorchBaseConfig):
    asian_support: bool = False
    lowercase: bool = True
    no_punctuation: bool = False
    normalize: bool = False
    return_sentence_level_score: bool = False


class TorchMetTschuprowsTConfig(TorchBaseConfig):
    bias_correction: bool = True
    nan_replace_value: OptFloat = 0.0
    nan_strategy: str = 'replace'
    num_classes: int


class TorchMetTweedieDevianceScoreConfig(TorchBaseConfig):
    power: float = 0.0


class TorchMetUniversalImageQualityIndexConfig(TorchBaseConfig, TMReduct):
    kernel_size: tuple[int, ...] = (11, 11)
    sigma: tuple[float, ...] = (1.5, 1.5)
