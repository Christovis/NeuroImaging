import numpy as np
from keras import backend as K

smooth_default = 1.

def dice_coef(y_true, y_pred, smooth = smooth_default,
              per_batch = True):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        sdc = (2.*intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return sdc
    else:
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f,
                              axis=1,
                              keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + \
                K.sum(y_pred_f, axis=1, keepdims=True) + \
                smooth
        return K.mean(intersec / union)


def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
