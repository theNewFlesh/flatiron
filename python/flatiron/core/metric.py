import tensorflow as tf
import tensorflow.keras.backend as tfkb
# ------------------------------------------------------------------------------


def intersection_over_union(y_true, y_pred, smooth=1.0):
    # type: (tf.Tensor, tf.Tensor, float) -> tf.Tensor
    r'''
    Intersection over union metric.

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
                intersection & \rightarrow \color{red2} I
                    (y, \hat{y}) && = \sum{(y * \hat{y})} 
                \\
                union & \rightarrow \color{green2} U
                    (y, \hat{y}) && = \sum{(y + \hat{y})} - I(y, \hat{y})
                \\
                \text{smoothing factor} & \rightarrow \color{blue2} S 
                \\ 
                \text{expansion} & \rightarrow 
                    \color{cyan2} IOU(y, \hat{y}, S) && = 
                    \frac{
                        \color{red2} \sum{(y * \hat{y})} 
                        \color{white} + \color{blue2} S
                    }{
                        \color{green2} \sum{(y + \hat{y})} - \sum{(y * \hat{y})} 
                        \color{white} + \color{blue2} S
                    } 
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
