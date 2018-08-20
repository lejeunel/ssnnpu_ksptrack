from keras import backend as K
import keras.losses as kloss

auxiliary_alpha = 0


def mean_squared_error_gaze(y_true, y_pred):
    """
    Calculate mean squared error multiplied by the last channel of y_true.
    If y_pred is of size (?, 224, 174, 3), then y_true must be (?, 224, 174, 4)

    G = y_true[:,:,:,-1:]
    y = y_true[:,:,:,:-1]

                 +----                      2
                 \      |                  |
    Loss  =  G    +     | y    - y_pred    |
        ij    ij /      |  ijc         ijc |
                 +----
                   c

    :param y_true: labels
    :param y_pred: predictions
    :return:       per pixel loss
    """
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value, sh[2]._value, sh[3]._value + 1))
    gauss = K.repeat_elements(y_true[..., -1:], y_pred.get_shape()[-1], len(y_pred.get_shape()) - 1)
    return K.mean(gauss * K.square(y_pred - y_true[..., :-1]), axis=-1)


def mean_absolute_error_gaze(y_true, y_pred):
    """
    Calculate mean absolute error multiplied by the last channel of y_true.
    If y_pred is of size (?, 224, 174, 3), then y_true must be (?, 224, 174, 4)

    G = y_true[:,:,:,-1:]
    y = y_true[:,:,:,:-1]

                 +----
                 \      |                  |
    Loss  =  G    +     | y    - y_pred    |
        ij    ij /      |  ijc         ijc |
                 +----
                   c

    :param y_true: labels
    :param y_pred: predictions
    :return:       per pixel loss
    """
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value, sh[2]._value, sh[3]._value + 1))
    gauss = K.repeat_elements(y_true[..., -1:], y_pred.get_shape()[-1], len(y_pred.get_shape()) - 1)
    return K.mean(gauss * K.abs(y_pred - y_true[..., :-1]), axis=-1)


def bce_auxiliary(y_true, y_pred):
    """
    Calculate mean absolute error multiplied by the last channel of y_true.
    If y_pred is of size (?, 224, 174, 3), then y_true must be (?, 224, 174, 4)

    G = y_true[:,:,:,-1:]
    y = y_true[:,:,:,:-1]

                          N-1
                        +----  +----
                     1  \      \       | ^              |                  | ^              |
    Loss  =  alpha * - * +      +   BCE| G(t-d), G(t-d) | + (1-alpha) * BCE| G(t+1), G(t+1) |
        ij           N  /      /       |  ijc     ijc   |                  |                |
                        +----  +----
                         d=0     c

    :param y_true: labels
    :param y_pred: predictions
    :return:       per pixel loss
    """
    sh = y_pred.get_shape()
    y_true.set_shape((sh[0]._value, sh[1]._value, sh[2]._value, sh[3]._value, sh[4]._value))

    n = sh[1]._value - 1
    aux_loss = 0
    for i in range(n):
        aux_loss += kloss.binary_crossentropy(y_true[:, i, ...], y_pred[:, i, ...])
    aux_loss /= n

    rec_loss = kloss.binary_crossentropy(y_true[:, n, ...], y_pred[:, n, ...])

    alpha = get_auxiliary_param()
    loss = alpha * aux_loss + (1-alpha)*rec_loss
    return loss


def set_auxiliary_param(alpha):
    global auxiliary_alpha
    auxiliary_alpha = 0.1

def get_auxiliary_param():
    return auxiliary_alpha
