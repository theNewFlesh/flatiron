import tensorflow as tf
import tensorflow.keras.backend as tfkb
# ------------------------------------------------------------------------------


def intersection_over_union(y_true, y_pred, smooth=1.0):
    # type: (tf.Tensor, tf.Tensor, float) -> tf.Tensor
    r'''
    Intersection over union metric.

    .. math::
        :nowrap:

            \begin{alignat*}{3}
                \definecolor{cyan2}{rgb}{0.71, 0.93, 0.95}
                \definecolor{red2}{rgb}{0.87, 0.58, 0.56}
                \color{red2} \text{intersection over union} & \rightarrow
                    \color{cyan2} IOU(y, \hat{y}, S) && = 
                    \frac{I + S}{U + S}
                \\
                \\
                & && = \frac{\sum{(y * \hat{y})} + S}
                    {\sum{y} + \sum{\hat{y}} - \sum{(y * \hat{y})} + S}
                \\
                \\
                \color{red2} intersection & \rightarrow
                    \color{cyan2} I(y, \hat{y}) && = \sum{(y * \hat{y})}
                \\
                \color{red2} union & \rightarrow
                    \color{cyan2} U(y, \hat{y}) && = 
                    \sum{y} + \sum{\hat{y}} - I(y, \hat{y})
                \\
                \color{red2} \text{smoothing factor} & \rightarrow
                    \color{cyan2} S
            \end{alignat*}

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
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
