from typing import Union
import numpy
import tensorflow as tf

import tensorflow.keras.backend as tfb

Arraylike = Union[numpy.ndarray, tf.Tensor]
# ------------------------------------------------------------------------------


def jaccards_loss(y_true, y_pred, smooth=100):
    # type: (Arraylike, Arraylike, int) -> tf.Tensor
    '''
    Jaccard's loss is usefull for unbalanced datasets. This has been shifted so
    it converges on 0 and is smoothed to avoid exploding or disappearing
    gradient.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
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
    '''
    Dice loss function with smoothing factor to prevent exploding
    or vanishing gradients.

    Dice = D(y, yhat, s) -> y A yhat

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
