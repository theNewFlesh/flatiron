from typing import Union
from flatiron.core.types import OptBool, Ints, OptInt, OptInts, OptStr
from flatiron.core.types import Floats, OptFloat, OptListFloat, OptPairFloat

import pydantic as pyd
# ------------------------------------------------------------------------------


# BASE--------------------------------------------------------------------------
class TorchBaseConfig(pyd.BaseModel):
    name: str


# FRAMEWORK---------------------------------------------------------------------
class TorchFramework(pyd.BaseModel):
    '''
    Configuration for calls to torch train function.

    Attributes:
        name (str): Framework name. Default: 'torch'.
        device (str, optional): Hardware device. Default: 'cuda'.
    '''
    name: str = 'torch'
    device: str = 'cuda'


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
    data_range: OptPairFloat = None


# OPTIMIZER---------------------------------------------------------------------
class TorchOptASGD(TorchOptBaseConfig, TCap, TDecay, TDiff, TFor, TMax):
    alpha: float = 0.75
    lambd: float = 0.0001
    t0: float = 1000000.0


class TorchOptAdadelta(TorchOptBaseConfig, TGroup1):
    rho: float = 0.9


class TorchOptAdafactor(TorchOptBaseConfig, TDecay, TEps, TFor, TMax):
    beta2_decay: float = -0.8
    clipping_threshold: float = 1.0  # convert to d


class TorchOptAdagrad(TorchOptBaseConfig, TDecay, TDiff, TEps, TFor, TMax):
    fused: OptBool = None
    initial_accumulator_value: float = 0
    lr_decay: float = 0


class TorchOptAdam(TorchOptBaseConfig, TGroup1, TBeta):
    amsgrad: bool = False
    fused: OptBool = None


class TorchOptAdamW(TorchOptBaseConfig, TGroup1, TBeta):
    amsgrad: bool = False
    fused: OptBool = None


class TorchOptAdamax(TorchOptBaseConfig, TGroup1, TBeta):
    pass


class TorchOptLBFGS(TorchOptBaseConfig):
    history_size: int = 100
    line_search_fn: OptStr = None
    max_eval: OptInt = None
    max_iter: int = 20
    tolerance_change: float = 1e-09
    tolerance_grad: float = 1e-07


class TorchOptNAdam(TorchOptBaseConfig, TGroup1, TBeta):
    momentum_decay: float = 0.004


class TorchOptRAdam(TorchOptBaseConfig, TGroup1, TBeta):
    pass


class TorchOptRMSprop(TorchOptBaseConfig, TGroup1):
    alpha: float = 0.99
    centered: bool = False
    momentum: float = 0


class TorchOptRprop(TorchOptBaseConfig, TCap, TDiff, TFor, TMax):
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-06, 50)


class TorchOptSGD(TorchOptBaseConfig, TDecay, TDiff, TFor, TMax):
    dampening: float = 0
    fused: OptBool = None
    momentum: float = 0
    nesterov: bool = False


class TorchOptSparseAdam(TorchOptBaseConfig, TEps, TMax, TBeta):
    pass


# LOSS--------------------------------------------------------------------------
class TorchLossBCELoss(TorchBaseConfig, TGroup2):
    pass


class TorchLossBCEWithLogitsLoss(TorchBaseConfig, TGroup2):
    pass


class TorchLossCTCLoss(TorchBaseConfig, TReduct):
    blank: int = 0
    zero_infinity: bool = False


class TorchLossCosineEmbeddingLoss(TorchBaseConfig, TGroup3):
    pass


class TorchLossCrossEntropyLoss(TorchBaseConfig, TGroup2):
    ignore_index: int = -100
    label_smoothing: float = 0.0


class TorchLossGaussianNLLLoss(TorchBaseConfig, TEps, TReduct):
    full: bool = False


class TorchLossHingeEmbeddingLoss(TorchBaseConfig, TGroup3):
    pass


class TorchLossHuberLoss(TorchBaseConfig, TReduct):
    delta: float = 1.0


class TorchLossKLDivLoss(TorchBaseConfig, TGroup2):
    log_target: bool = False


class TorchLossL1Loss(TorchBaseConfig, TGroup2):
    pass


class TorchLossMSELoss(TorchBaseConfig, TGroup2):
    pass


class TorchLossMarginRankingLoss(TorchBaseConfig, TGroup3):
    pass


class TorchLossMultiLabelMarginLoss(TorchBaseConfig, TGroup2):
    pass


class TorchLossMultiLabelSoftMarginLoss(TorchBaseConfig, TGroup2):
    pass


class TorchLossMultiMarginLoss(TorchBaseConfig, TGroup3):
    exponent: int = 1  # convert to p


class TorchLossNLLLoss(TorchBaseConfig, TGroup2):
    ignore_index: int = -100


class TorchLossPairwiseDistance(TorchBaseConfig, TEps):
    keepdim: bool = False
    norm_degree: float = 2.0  # convert to p


class TorchLossPoissonNLLLoss(TorchBaseConfig, TEps, TGroup2):
    full: bool = False
    log_input: bool = True


class TorchLossSmoothL1Loss(TorchBaseConfig, TGroup2):
    beta: float = 1.0


class TorchLossSoftMarginLoss(TorchBaseConfig, TGroup2):
    pass


class TorchLossTripletMarginLoss(TorchBaseConfig, TEps, TGroup3):
    norm_degree: float = 2.0  # convert to p
    swap: bool = False


class TorchLossTripletMarginWithDistanceLoss(TorchBaseConfig, TMarg, TReduct):
    swap: bool = False


# METRICS-----------------------------------------------------------------------
class TorchMetricBLEUScore(TorchBaseConfig):
    n_gram: int = 4
    smooth: bool = False
    weights: OptListFloat = None


class TorchMetricCHRFScore(TorchBaseConfig):
    beta: float = 2.0
    lowercase: bool = False
    n_char_order: int = 6
    n_word_order: int = 2
    return_sentence_level_score: bool = False
    whitespace: bool = False


class TorchMetricCatMetric(TorchBaseConfig, TNan):
    pass


class TorchMetricConcordanceCorrCoef(TorchBaseConfig, TOut):
    pass


class TorchMetricCosineSimilarity(TorchBaseConfig):
    reduction: str = 'sum'


class TorchMetricCramersV(TorchBaseConfig, TCls, TNan):
    bias_correction: bool = True
    nan_replace_value: OptFloat = 0.0


class TorchMetricCriticalSuccessIndex(TorchBaseConfig):
    keep_sequence_dim: OptInt = None
    threshold: float


class TorchMetricDice(TorchBaseConfig, TCls, TInd, TTopK):
    average: OptStr = 'micro'
    mdmc_average: OptStr = 'global'
    multiclass: OptBool = None
    threshold: float = 0.5
    zero_division: int = 0


class TorchMetricErrorRelativeGlobalDimensionlessSynthesis(TorchBaseConfig, TMReduct):
    ratio: float = 4


class TorchMetricExplainedVariance(TorchBaseConfig):
    multioutput: str = 'uniform_average'


class TorchMetricExtendedEditDistance(TorchBaseConfig):
    alpha: float = 2.0
    deletion: float = 0.2
    insertion: float = 1.0
    language: str = 'en'
    return_sentence_level_score: bool = False
    rho: float = 0.3


class TorchMetricFleissKappa(TorchBaseConfig):
    mode: str = 'counts'


class TorchMetricKLDivergence(TorchBaseConfig):
    log_prob: bool = False
    reduction: str = 'mean'


class TorchMetricKendallRankCorrCoef(TorchBaseConfig, TOut):
    alternative: OptStr = 'two-sided'
    t_test: bool = False
    variant: str = 'b'


class TorchMetricLogCoshError(TorchBaseConfig, TOut):
    pass


class TorchMetricMaxMetric(TorchBaseConfig, TNan):
    pass


class TorchMetricMeanAbsoluteError(TorchBaseConfig, TOut):
    pass


class TorchMetricMeanMetric(TorchBaseConfig, TNan):
    pass


class TorchMetricMeanSquaredError(TorchBaseConfig, TOut):
    squared: bool = True


class TorchMetricMinMetric(TorchBaseConfig, TNan):
    pass


class TorchMetricMinkowskiDistance(TorchBaseConfig):
    p: float


class TorchMetricModifiedPanopticQuality(TorchBaseConfig):
    allow_unknown_preds_category: bool = False
    stuffs: list[int]
    things: list[int]


class TorchMetricMultiScaleStructuralSimilarityIndexMeasure(TorchBaseConfig, TMReduct, TDate):
    betas: tuple = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    gaussian_kernel: bool = True
    k1: float = 0.01
    k2: float = 0.03
    kernel_size: Ints = 11
    normalize: str = 'relu'
    sigma: Floats = 1.5


class TorchMetricNormalizedRootMeanSquaredError(TorchBaseConfig, TOut):
    normalization: str = 'mean'


class TorchMetricPanopticQuality(TorchBaseConfig):
    allow_unknown_preds_category: bool = False
    stuffs: list[int]
    things: list[int]


class TorchMetricPeakSignalNoiseRatio(TorchBaseConfig, TMReduct, TDate):
    base: float = 10.0
    dim: OptInts = None


class TorchMetricPearsonCorrCoef(TorchBaseConfig, TOut):
    pass


class TorchMetricPearsonsContingencyCoefficient(TorchBaseConfig):
    nan_replace_value: OptFloat = 0.0
    nan_strategy: str = 'replace'
    num_classes: int


class TorchMetricPermutationInvariantTraining(TorchBaseConfig):
    eval_func: str = 'max'
    mode: str = 'speaker-wise'


class TorchMetricPerplexity(TorchBaseConfig, TInd):
    pass


class TorchMetricR2Score(TorchBaseConfig):
    adjusted: int = 0
    multioutput: str = 'uniform_average'


class TorchMetricRelativeAverageSpectralError(TorchBaseConfig):
    window_size: int = 8


class TorchMetricRelativeSquaredError(TorchBaseConfig, TOut):
    squared: bool = True


class TorchMetricRetrievalFallOut(TorchBaseConfig, TInd, TTopK):
    empty_target_action: str = 'pos'


class TorchMetricRetrievalHitRate(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetricRetrievalMAP(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetricRetrievalMRR(TorchBaseConfig, TAct, TInd):
    pass


class TorchMetricRetrievalNormalizedDCG(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetricRetrievalPrecision(TorchBaseConfig, TAct, TInd, TTopK):
    adaptive_k: bool = False


class TorchMetricRetrievalPrecisionRecallCurve(TorchBaseConfig, TInd):
    adaptive_k: bool = False
    max_k: OptInt = None


class TorchMetricRetrievalRPrecision(TorchBaseConfig, TAct, TInd):
    pass


class TorchMetricRetrievalRecall(TorchBaseConfig, TAct, TInd, TTopK):
    pass


class TorchMetricRetrievalRecallAtFixedPrecision(TorchBaseConfig, TAct, TInd):
    adaptive_k: bool = False
    max_k: OptInt = None
    min_precision: float = 0.0


class TorchMetricRootMeanSquaredErrorUsingSlidingWindow(TorchBaseConfig):
    window_size: int = 8


class TorchMetricRunningMean(TorchBaseConfig, TNan):
    window: int = 5


class TorchMetricRunningSum(TorchBaseConfig, TNan):
    window: int = 5


class TorchMetricSacreBLEUScore(TorchBaseConfig):
    lowercase: bool = False
    n_gram: int = 4
    smooth: bool = False
    tokenize: str = '13a'
    weights: OptListFloat = None


class TorchMetricScaleInvariantSignalDistortionRatio(TorchBaseConfig):
    zero_mean: bool = False


class TorchMetricSignalDistortionRatio(TorchBaseConfig):
    filter_length: int = 512
    load_diag: OptFloat = None
    use_cg_iter: OptInt = None
    zero_mean: bool = False


class TorchMetricSignalNoiseRatio(TorchBaseConfig):
    zero_mean: bool = False


class TorchMetricSpearmanCorrCoef(TorchBaseConfig, TOut):
    pass


class TorchMetricSpectralAngleMapper(TorchBaseConfig, TMReduct):
    pass


class TorchMetricSpectralDistortionIndex(TorchBaseConfig, TMReduct):
    p: int = 1


class TorchMetricStructuralSimilarityIndexMeasure(TorchBaseConfig, TMReduct):
    data_range: OptPairFloat = None
    gaussian_kernel: bool = True
    k1: float = 0.01
    k2: float = 0.03
    kernel_size: Ints = 11
    return_contrast_sensitivity: bool = False
    return_full_image: bool = False
    sigma: Floats = 1.5


class TorchMetricSumMetric(TorchBaseConfig, TNan):
    pass


class TorchMetricTheilsU(TorchBaseConfig):
    nan_replace_value: OptFloat = 0.0
    nan_strategy: str = 'replace'
    num_classes: int


class TorchMetricTotalVariation(TorchBaseConfig):
    reduction: str = 'sum'


class TorchMetricTranslationEditRate(TorchBaseConfig):
    asian_support: bool = False
    lowercase: bool = True
    no_punctuation: bool = False
    normalize: bool = False
    return_sentence_level_score: bool = False


class TorchMetricTschuprowsT(TorchBaseConfig):
    bias_correction: bool = True
    nan_replace_value: OptFloat = 0.0
    nan_strategy: str = 'replace'
    num_classes: int


class TorchMetricTweedieDevianceScore(TorchBaseConfig):
    power: float = 0.0


class TorchMetricUniversalImageQualityIndex(TorchBaseConfig, TMReduct):
    kernel_size: tuple[int, ...] = (11, 11)
    sigma: tuple[float, ...] = (1.5, 1.5)
