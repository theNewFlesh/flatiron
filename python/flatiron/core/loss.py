from typing import Union
import numpy
import tensorflow as tf

import tensorflow.keras.backend as tfb

Arraylike = Union[numpy.ndarray, tf.Tensor]
# ------------------------------------------------------------------------------


def jaccards_loss(y_true, y_pred, smooth=100):
    # type: (Arraylike, Arraylike, int) -> tf.Tensor
    r'''
    Jaccard's loss is usefull for unbalanced datasets. This has been shifted so
    it converges on 0 and is smoothed to avoid exploding or disappearing
    gradients.

    .. math::
        :nowrap:

            \begin{align}
                intersection & \rightarrow I(y, \hat{y}) = \sum_{i=0}^{n}{|y_i * \hat{y}_i|} \\
                union & \rightarrow U(y, \hat{y}) = \sum_{i=0}^{n}{|y_i| + |\hat{y}_i|} \\
                smooth & \rightarrow S \\
                loss & \rightarrow L_{jaccard}(y, \hat{y}, S) = 1 - \frac{I + S}{U - I + S} + S \\
            \end{align}

    See: https://en.wikipedia.org/wiki/Jaccard_index

    Args:
        y_true (NDArray or Tensor): Ground truth labels.
        y_pred (NDArray or Tensor): Predicted labels.
        smooth (int, optional): Smoothing factor. Default: 100.

    Returns:
        tf.Tensor: Loss function.
    '''
    intersection = tfb.sum(tfb.abs(y_true * y_pred), axis=-1)
    union = tfb.sum(tfb.abs(y_true) + tfb.abs(y_pred), axis=-1)
    jaccard = (intersection + smooth) / (union - intersection + smooth)
    loss = (1 - jaccard) * smooth
    return loss


def dice_loss(y_true, y_pred, smooth=1):
    # type: (Arraylike, Arraylike, int) -> tf.Tensor
    r'''
    Dice loss function with smoothing factor to prevent exploding or vanishing
    gradients.

    .. math::
        :nowrap:

            \begin{align}
                intersection & \rightarrow I(y, \hat{y}) = \sum_{i=0}^{n}{|y_i * \hat{y}_i|} \\
                union & \rightarrow U(y, \hat{y}) = \sum_{i=0}^{n}{|y_i| + |\hat{y}_i|} \\
                smooth & \rightarrow S \\
                loss & \rightarrow L_{dice}(y, \hat{y}, S) = 1 - \frac{2 * I + S}{U + S} \\
            \end{align}

    See: https://cnvrg.io/semantic-segmentation/

    Args:
        y_true (NDArray or Tensor): Ground truth labels.
        y_pred (NDArray or Tensor): Predicted labels.
        smooth (int, optional): Smoothing factor. Default: 1.

    Returns:
        tf.Tensor: Loss function.
    '''
    intersection = tfb.sum(tfb.abs(y_true * y_pred), axis=-1)
    union = tfb.sum(tfb.abs(y_true) + tfb.abs(y_pred), axis=-1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss
