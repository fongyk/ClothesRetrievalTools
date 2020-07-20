import torch
import numpy as np
from sklearn.preprocessing import normalize as sknormalize

def typename(x):
    return type(x).__module__


def tonumpy(x):
    if typename(x) == torch.__name__:
        return x.cpu().numpy()
    else:
        return x


def matmul(A, B):
    if typename(A) == np.__name__:
        B = tonumpy(B)
        scores = np.dot(A, B.T)
    elif typename(B) == torch.__name__:
        scores = torch.matmul(A, B.t()).cpu().numpy()
    else:
        raise TypeError("matrices must be either numpy or torch type")
    return scores

def euclidean_matrix(x, y):
    '''
    calculate euclidean distances between x and y.
    x: m x d, numpy.ndarray
    y: n x d, numpy.ndarray
    note: dist(u, v) = sqrt(u^2 + v^2 - 2uv)
    '''
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    m, n = x.shape[0], y.shape[0]

    x_norm = np.sum(np.power(x, 2), axis=1, keepdims=True)
    x_norm = np.tile(x_norm, (1, n)) ## m x n

    y_norm = np.sum(np.power(y, 2), axis=1, keepdims=True)
    y_norm = np.tile(y_norm.T, (m, 1)) ## m x n

    sim = np.dot(x, y.T)

    dis_mat = np.sqrt(x_norm + y_norm - 2 * sim)

    return dis_mat

def normalize(x, norm='l2', axis=1, copy=True):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and x.ndim == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), norm='l2', axis=1, copy=copy))
    else:
        return sknormalize(x, norm='l2', axis=1, copy=copy)
