from typing import Any, Callable, Union  # noqa F401
import numpy

import tensorflow as tf
import tensorflow.keras.backend as tfb

import flatiron.core.tools as fict

Arraylike = Union[numpy.ndarray, tf.Tensor]
# ------------------------------------------------------------------------------


def get(name):
    # type: (str) -> Callable[[Any], Any]
    '''
    Get function from this module.

    Args:
        name (str): Function name.

    Returns:
        function: Module function.
    '''
    return fict.get_module_function(name, __name__)
# ------------------------------------------------------------------------------


def jaccard_loss(y_true, y_pred, smooth=100):
    # type: (Arraylike, Arraylike, int) -> tf.Tensor
    r'''
    Jaccard's loss is usefull for unbalanced datasets. This has been shifted so
    it converges on 0 and is smoothed to avoid exploding or disappearing
    gradients.

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

                \color{cyan2} L_{jacc}(y, \hat{y}, S) && =
                    (1 - \frac{I + S}{U - I + S}) S
            \end{alignat*}

    Terms:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                intersection & \rightarrow \color{red2}
                    I(y, \hat{y}) && = \sum{|y_i * \hat{y_i}|}
                \\
                union & \rightarrow \color{green2}
                    U(y, \hat{y}) && = \sum{(|y_i| + |\hat{y_i}|)}
                \\
                \text{smoothing factor} & \rightarrow \color{blue2} S &&
                \\
                \text{expansion} & \rightarrow
                    \color{cyan2} L_{jacc}(y, \hat{y}, S) && =
                        (1 - \frac{
                            \color{red2} \sum{|y_i * \hat{y_i}|}
                            \color{white} + \color{blue2} S
                        }{
                            \color{green2} \sum{(|y_i| + |\hat{y_i}|)}
                            \color{white} -
                            \color{red2} \sum{|y_i * \hat{y_i}|}
                            \color{white} + \color{blue2} S
                        }) \color{blue2} S
            \end{alignat*}

    Args:
        y_true (NDArray or Tensor): Ground truth labels.
        y_pred (NDArray or Tensor): Predicted labels.
        smooth (int, optional): Smoothing factor. Default: 100.

    Returns:
        tf.Tensor: Loss function.
    '''
    i = tfb.sum(tfb.abs(y_true * y_pred), axis=-1)
    u = tfb.sum(tfb.abs(y_true) + tfb.abs(y_pred), axis=-1)
    jacc = (i + smooth) / (u - i + smooth)
    loss = (1 - jacc) * smooth
    return loss


def dice_loss(y_true, y_pred, smooth=1):
    # type: (Arraylike, Arraylike, int) -> tf.Tensor
    r'''
    Dice loss function with smoothing factor to prevent exploding or vanishing
    gradients.

    See: https://cnvrg.io/semantic-segmentation

    Equation:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                \definecolor{blue2}{rgb}{0.58, 0.71, 0.9}
                \definecolor{cyan2}{rgb}{0.71, 0.93, 0.95}
                \definecolor{green2}{rgb}{0.63, 0.82, 0.48}
                \definecolor{light1}{rgb}{0.64, 0.64, 0.64}
                \definecolor{red2}{rgb}{0.87, 0.58, 0.56}

                \color{cyan2} L_{dice}(y, \hat{y}, S) && =
                    1 - \frac{2 * I + S}{U + S}
            \end{alignat*}

    Terms:

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                intersection & \rightarrow \color{red2}
                    I(y, \hat{y}) && = \sum{|y_i * \hat{y_i}|}
                \\
                union & \rightarrow \color{green2}
                    U(y, \hat{y}) && = \sum{(|y_i| + |\hat{y_i}|)}
                \\
                \text{smoothing factor} & \rightarrow \color{blue2} S &&
                \\
                \text{expansion} & \rightarrow
                    \color{cyan2} L_{dice}(y, \hat{y}, S) && =
                        1 - \frac{
                            2 *
                            \color{red2} \sum{|y_i * \hat{y_i}|}
                            \color{white} + \color{blue2} S
                        }{
                            \color{green2} \sum{(|y_i| + |\hat{y_i}|)}
                            \color{white} + \color{blue2} S
                        }
            \end{alignat*}

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
