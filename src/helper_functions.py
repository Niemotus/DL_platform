import numpy as np
import theano.tensor as T

def categorical_crossentropy(prediction, truth):
    # prediction: label probability prediction, 4D tensor
    # truth: ground truth label map, 1-hot representation, 4D tensor
    return - T.mean(T.sum(truth * T.log(prediction+1e-8), axis=1))

def sorenson_dice(prediction, truth, k):
    return 1-2 * T.sum(prediction[:,k,:,:] * truth[:,k,:,:]) / T.cast(T.sum(prediction[:,k,:,:]) + T.sum(truth[:,k,:,:]) + 1e-8, 'float32')


def categorical_dice_theano(pred, gt_label, k):
    # Dice overlap metric for label value k
    A = T.eq(pred, k)
    B = T.eq(gt_label, k)
    return 2 * T.sum(A*B) / T.cast(T.sum(A)+T.sum(B), 'float32')


def categorical_dice(pred, gt_label, k):
    # Dice overlap metric for label value k
    A = pred == k
    B = gt_label == k
    return 2 * np.sum(A*B) / (np.sum(A)+np.sum(B)).astype('float32')


def calculate_dice(prediction, truth, k):
    # Dice overlap metric for label value k
    A = (np.argmax(prediction, axis=0))==k
    B = truth==k
    A = A.astype('float32')
    B = B.astype('float32')
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))
