from typing import Union
import numpy

import tensorflow as tf
import tensorflow.keras.backend as tfkb

Arraylike = Union[numpy.ndarray, tf.Tensor]
# ------------------------------------------------------------------------------


def intersection_over_union(y_true, y_pred, smooth=1.0):
    # type: (Arraylike, Arraylike, float) -> tf.Tensor
    r'''
    Intersection over union metric.

    See: https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef

    Equation:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                \definecolor{blue2}{rgb}{0.58, 0.71, 0.9}
                \definecolor{cyan2}{rgb}{0.71, 0.93, 0.95}
                \definecolor{green2}{rgb}{0.63, 0.82, 0.48}
                \definecolor{light1}{rgb}{0.64, 0.64, 0.64}
                \definecolor{red2}{rgb}{0.87, 0.58, 0.56}

                \color{cyan2} IOU (y, \hat{y}, S) && = \frac{I + S}{U + S}
            \end{alignat*}

    Terms:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                intersection & \rightarrow \color{red2} 
                    I(y, \hat{y}) && = \sum{(y_i * \hat{y_i})} 
                \\
                union & \rightarrow \color{green2} 
                    U(y, \hat{y}) && = \sum{(y_i + \hat{y_i})} - I(y_i, \hat{y_i})
                \\
                \text{smoothing factor} & \rightarrow \color{blue2} S 
                \\ 
                \text{expansion} & \rightarrow 
                    \color{cyan2} IOU(y, \hat{y}, S) && = 
                    \frac{
                        \color{red2} \sum{(y_i * \hat{y_i})} 
                        \color{white} + \color{blue2} S
                    }{
                        \color{green2} \sum{(y_i + \hat{y_i})} - \sum{(y_i * \hat{y_i})} 
                        \color{white} + \color{blue2} S
                    } 
            \end{alignat*}

    Args:
        y_true (NDArray or Tensor): True labels.
        y_pred (NDArray or Tensor): Predicted labels.
        smooth (float, optional): Smoothing factor. Default: 1.0

    Returns:
        tf.Tensor: IOU metric.
    '''
    yt = tfkb.flatten(y_true)
    yp = tfkb.flatten(y_pred)
    i = tfkb.sum(yt * yp)
    u = tfkb.sum(yt) + tfkb.sum(yp) - i
    iou = (i + smooth) / (u + smooth)
    return iou


def jaccard(y_true, y_pred):
    # type: (Arraylike, Arraylike) -> tf.Tensor
    r'''
    Jaccard metric.

    See: https://en.wikipedia.org/wiki/Jaccard_index

    Equation:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                \definecolor{blue2}{rgb}{0.58, 0.71, 0.9}
                \definecolor{cyan2}{rgb}{0.71, 0.93, 0.95}
                \definecolor{green2}{rgb}{0.63, 0.82, 0.48}
                \definecolor{light1}{rgb}{0.64, 0.64, 0.64}
                \definecolor{red2}{rgb}{0.87, 0.58, 0.56}

                \color{cyan2} Jacc(y, \hat{y}) && = 
                    \frac{1}{N} \sum_{i=0}^{N} \frac{I + 1}{U + 1}
            \end{alignat*}

    Terms:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                intersection & \rightarrow \color{red2} 
                    I(y, \hat{y}) && = \sum{(y_i * \hat{y_i})} 
                \\
                union & \rightarrow \color{green2} 
                    U(y, \hat{y}) && = \sum{(y_i + \hat{y_i})} - I(y_i, \hat{y_i})
                \\
                \text{expansion} & \rightarrow 
                    \color{cyan2} Jacc(y, \hat{y}) && = 
                    \frac{1}{N} \sum_{i=0}^{N}
                    \frac{
                        \color{red2} \sum{(y_i * \hat{y_i})} 
                        \color{white} + 1
                    }{
                        \color{green2} \sum{(y_i + \hat{y_i})} - \sum{(y_i * \hat{y_i})} 
                        \color{white} + 1
                    } 
            \end{alignat*}

    Args:
        y_true (NDArray or Tensor): True labels.
        y_pred (NDArray or Tensor): Predicted labels.

    Returns:
        tf.Tensor: Jaccard metric.
    '''
    i = tfkb.sum(y_true * y_pred)
    u = tfkb.sum(y_true + y_pred) - i
    jacc = (i + 1.0) / (u + 1.0)
    jacc = tfkb.mean(jacc)
    return jacc


def dice(y_true, y_pred, smooth=1.0):
    # type: (Arraylike, Arraylike, float) -> tf.Tensor
    r'''
    Dice metric.

    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Equation:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                \definecolor{blue2}{rgb}{0.58, 0.71, 0.9}
                \definecolor{cyan2}{rgb}{0.71, 0.93, 0.95}
                \definecolor{green2}{rgb}{0.63, 0.82, 0.48}
                \definecolor{light1}{rgb}{0.64, 0.64, 0.64}
                \definecolor{red2}{rgb}{0.87, 0.58, 0.56}

                \color{cyan2} Dice(y, \hat{y}) && = \frac{2 * I + S}{U_s + S}
            \end{alignat*}

    Terms:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                intersection & \rightarrow \color{red2} 
                    I(y, \hat{y}) && = \sum{(y_i * \hat{y_i})} 
                \\
                \text{union} & \rightarrow \color{green2} 
                    U_s(y, \hat{y}) && = \sum{(y_i + \hat{y_i})}
                \\
                \text{smoothing factor} & \rightarrow \color{blue2} S 
                \\ 
                \text{expansion} & \rightarrow 
                    \color{cyan2} Dice(y, \hat{y}, S) && = 
                    \frac{ 
                        \color{white} 2 * \color{red2} \sum{(y_i * \hat{y_i})} 
                        \color{white} + \color{blue2} S
                    }{
                        \color{green2} \sum{(y_i + \hat{y_i})}
                        \color{white} + \color{blue2} S
                    } 
            \end{alignat*}

    Args:
        y_true (NDArray or Tensor): True labels.
        y_pred (NDArray or Tensor): Predicted labels.
        smooth (float, optional): Smoothing factor. Default: 1.0

    Returns:
        tf.Tensor: Dice metric.
    '''
    yt = tfkb.flatten(y_true)
    yp = tfkb.flatten(y_pred)
    i = tfkb.sum(yt * yp)
    u = tfkb.sum(yt) + tfkb.sum(yp)
    dice = (2.0 * i + smooth) / (u + smooth)
    return dice
