from flatiron.core.types import OptInt, OptFloat, OptStr, OptFloats, OptListFloat

import pydantic as pyd
# ------------------------------------------------------------------------------


class TFBaseConfig(pyd.BaseModel):
    name: str


# OPTIMIZER-HELPERS-------------------------------------------------------------
class TFOptBaseConfig(TFBaseConfig):
    clipnorm: OptFloat = None
    clipvalue: OptFloat = None
    ema_momentum: float = 0.99
    ema_overwrite_frequency: OptInt = None
    global_clipnorm: OptFloat = None
    gradient_accumulation_steps: OptInt = None
    learning_rate: float = 0.001
    loss_scale_factor: OptFloat = None
    use_ema: bool = False
    # weight_decay: OptFloat = None


class TFEpsilon(pyd.BaseModel):
    epsilon: float = 1e-07


class TFBeta(pyd.BaseModel):
    beta_1: float = 0.9
    beta_2: float = 0.99


# LOSS-HELPERS------------------------------------------------------------------
class TFLossBaseConfig(TFBaseConfig):
    dtype: OptStr = None
    reduction: str = 'sum_over_batch_size'


class TFAxis(pyd.BaseModel):
    axis: int = -1


class TFLogits(pyd.BaseModel):
    from_logits: bool = False


# METRIC-HELPERS----------------------------------------------------------------
class TFMetricBaseConfig(TFBaseConfig):
    dtype: OptStr = None


class TFThresh(pyd.BaseModel):
    thresholds: OptFloats = None


class TFClsId(pyd.BaseModel):
    class_id: OptInt = None


class TFNumThresh(pyd.BaseModel):
    num_thresholds: int = 200


# OPTIMIZERS--------------------------------------------------------------------
class TFOptAdafactor(TFOptBaseConfig):
    beta_2_decay: float = -0.8
    clip_threshold: float = 1.0
    epsilon_1: float = 1e-30
    epsilon_2: float = 0.001
    relative_step: bool = True


class TFOptFtrl(TFOptBaseConfig):
    beta: float = 0.0
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    l2_shrinkage_regularization_strength: float = 0.0
    learning_rate_power: float = -0.5


class TFOptLion(TFOptBaseConfig, TFBeta):
    pass


class TFOptSGD(TFOptBaseConfig):
    momentum: float = 0.0
    nesterov: bool = False


class TFOptAdadelta(TFOptBaseConfig, TFEpsilon):
    rho: float = 0.95


class TFOptAdagrad(TFOptBaseConfig, TFEpsilon):
    initial_accumulator_value: float = 0.1


class TFOptAdam(TFOptBaseConfig, TFBeta, TFEpsilon):
    amsgrad: bool = False


class TFOptAdamW(TFOptBaseConfig, TFBeta, TFEpsilon):
    amsgrad: bool = False
    weight_decay: float = 0.004


class TFOptAdamax(TFOptBaseConfig, TFBeta, TFEpsilon):
    pass


# class TFOptLamb(TFOptBaseConfig, TFBeta, TFEpsilon):
#     pass


class TFOptNadam(TFOptBaseConfig, TFBeta, TFEpsilon):
    pass


class TFOptRMSprop(TFOptBaseConfig, TFEpsilon):
    centered: bool = False
    momentum: float = 0.0
    rho: float = 0.9


# LOSSES------------------------------------------------------------------------
class TFLossBinaryCrossentropy(TFLossBaseConfig, TFAxis, TFLogits):
    label_smoothing: float = 0.0


class TFLossBinaryFocalCrossentropy(TFLossBaseConfig, TFAxis, TFLogits):
    alpha: float = 0.25
    apply_class_balancing: bool = False
    gamma: float = 2.0
    label_smoothing: float = 0.0


class TFLossCategoricalCrossentropy(TFLossBaseConfig, TFAxis, TFLogits):
    label_smoothing: float = 0.0


class TFLossCategoricalFocalCrossentropy(TFLossBaseConfig, TFAxis, TFLogits):
    alpha: float = 0.25
    gamma: float = 2.0
    label_smoothing: float = 0.0


class TFLossCircle(TFLossBaseConfig):
    gamma: float = 80.0
    margin: float = 0.4
    remove_diagonal: bool = True


class TFLossCosineSimilarity(TFLossBaseConfig, TFAxis):
    pass


class TFLossDice(TFLossBaseConfig, TFAxis):
    pass


class TFLossHuber(TFLossBaseConfig):
    delta: float = 1.0


class TFLossSparseCategoricalCrossentropy(TFLossBaseConfig, TFLogits):
    ignore_class: OptInt = None


class TFLossTversky(TFLossBaseConfig, TFAxis):
    alpha: float = 0.5
    beta: float = 0.5


# METRICS-----------------------------------------------------------------------
class TFMetricAUC(TFMetricBaseConfig, TFNumThresh, TFThresh):
    curve: str = 'ROC'
    from_logits: bool = False
    label_weights: OptListFloat = None
    multi_label: bool = False
    num_labels: OptInt = None
    summation_method: str = 'interpolation'


class TFMetricAccuracy(TFMetricBaseConfig):
    pass


class TFMetricBinaryAccuracy(TFMetricBaseConfig):
    threshold: float = 0.5


class TFMetricBinaryCrossentropy(TFMetricBaseConfig):
    from_logits: bool = False
    label_smoothing: int = 0


class TFMetricBinaryIoU(TFMetricBaseConfig):
    target_class_ids: list[int] = [0, 1]
    threshold: float = 0.5


class TFMetricCategoricalAccuracy(TFMetricBaseConfig):
    pass


class TFMetricCategoricalCrossentropy(TFMetricBaseConfig, TFAxis):
    from_logits: bool = False
    label_smoothing: int = 0


class TFMetricCategoricalHinge(TFMetricBaseConfig):
    pass


class TFMetricConcordanceCorrelation(TFMetricBaseConfig, TFAxis):
    pass


class TFMetricCosineSimilarity(TFMetricBaseConfig, TFAxis):
    pass


class TFMetricF1Score(TFMetricBaseConfig):
    average: OptStr = None
    threshold: OptFloat = None


class TFMetricFBetaScore(TFMetricBaseConfig):
    average: OptStr = None
    beta: float = 1.0
    threshold: OptFloat = None


class TFMetricFalseNegatives(TFMetricBaseConfig, TFThresh):
    pass


class TFMetricFalsePositives(TFMetricBaseConfig, TFThresh):
    pass


class TFMetricHinge(TFMetricBaseConfig):
    pass


# class TFMetricIoU(TFMetricBaseConfig, TFAxis):
#     ignore_class: OptInt = None
#     num_classes: int
#     sparse_y_pred: bool = True
#     sparse_y_true: bool = True
#     target_class_ids: list[int]


class TFMetricKLDivergence(TFMetricBaseConfig):
    pass


class TFMetricLogCoshError(TFMetricBaseConfig):
    pass


class TFMetricMean(TFMetricBaseConfig):
    pass


class TFMetricMeanAbsoluteError(TFMetricBaseConfig):
    pass


class TFMetricMeanAbsolutePercentageError(TFMetricBaseConfig):
    pass


# class TFMetricMeanIoU(TFMetricBaseConfig, TFAxis):
#     ignore_class: OptInt = None
#     num_classes: int
#     sparse_y_pred: bool = True
#     sparse_y_true: bool = True


class TFMetricMeanSquaredError(TFMetricBaseConfig):
    pass


class TFMetricMeanSquaredLogarithmicError(TFMetricBaseConfig):
    pass


class TFMetricMetric(TFMetricBaseConfig):
    pass


# class TFMetricOneHotIoU(TFMetricBaseConfig, TFAxis):
#     ignore_class: OptInt = None
#     num_classes: int
#     sparse_y_pred: bool = False
#     target_class_ids: list[int]


# class TFMetricOneHotMeanIoU(TFMetricBaseConfig, TFAxis):
#     ignore_class: OptInt = None
#     num_classes: int
#     sparse_y_pred: bool = False


class TFMetricPearsonCorrelation(TFMetricBaseConfig, TFAxis):
    pass


class TFMetricPoisson(TFMetricBaseConfig):
    pass


class TFMetricPrecision(TFMetricBaseConfig, TFClsId, TFThresh):
    top_k: OptInt = None


# class TFMetricPrecisionAtRecall(TFMetricBaseConfig, TFClsId, TFNumThresh):
#     recall: float


class TFMetricR2Score(TFMetricBaseConfig):
    class_aggregation: str = 'uniform_average'
    num_regressors: int = 0


class TFMetricRecall(TFMetricBaseConfig, TFClsId, TFThresh):
    top_k: OptInt = None


# class TFMetricRecallAtPrecision(TFMetricBaseConfig, TFClsId, TFNumThresh):
#     precision: float


class TFMetricRootMeanSquaredError(TFMetricBaseConfig):
    pass


# class TFMetricSensitivityAtSpecificity(TFMetricBaseConfig, TFClsId, TFNumThresh):
#     specificity: float


class TFMetricSparseCategoricalAccuracy(TFMetricBaseConfig):
    pass


class TFMetricSparseCategoricalCrossentropy(TFMetricBaseConfig, TFAxis):
    from_logits: bool = False


class TFMetricSparseTopKCategoricalAccuracy(TFMetricBaseConfig):
    from_sorted_ids: bool = False
    k: int = 5


# class TFMetricSpecificityAtSensitivity(TFMetricBaseConfig, TFClsId, TFNumThresh):
#     sensitivity: float


class TFMetricSquaredHinge(TFMetricBaseConfig):
    pass


class TFMetricSum(TFMetricBaseConfig):
    pass


class TFMetricTopKCategoricalAccuracy(TFMetricBaseConfig):
    k: int = 5


class TFMetricTrueNegatives(TFMetricBaseConfig, TFThresh):
    pass


class TFMetricTruePositives(TFMetricBaseConfig, TFThresh):
    pass
