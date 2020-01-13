import numpy as np
import scipy.sparse as sp
from numba import njit

from pyzsl.utils.general import vec
from .common import _metric_doc, _dispatch, scores_to_topk, _INT_TYPES, \
    _FLOAT_TYPES, scores_to_ranks


@_metric_doc('Computes precision@k, that is a number of true labels in top k predictions, divided by k')
def precision_at_k(y_true, preds, k=1):
    return _precision_at_k(y_true, preds, k)


@_dispatch(np.ndarray, _INT_TYPES)
def _precision_at_k(y_true, preds, k):
    return _precision_at_k_topk(y_true.indices, y_true.indptr, preds)


@_dispatch(np.ndarray, _FLOAT_TYPES)
def _precision_at_k(y_true, preds, k):
    return precision_at_k(y_true, scores_to_ranks(preds, y_true), k)


@_dispatch(sp.csr_matrix)
def _precision_at_k(y_true: sp.csr_matrix, preds: sp.csr_matrix, k: int):

    sizes = vec((preds > 0).sum(1), dtype=np.float64)
    sizes[sizes == 0] = 1

    misses = vec((preds > k).sum(1), dtype=np.float64)
    hits   = sizes - misses

    ret = hits / k

    return np.mean(ret)


@njit
def _precision_at_k_topk(y_indices, y_indptr, preds):
    k   = preds.shape[1]
    ret = 0.0

    for i, (start, stop) in enumerate(zip(y_indptr, y_indptr[1:])):
        for idx in range(start, stop):

            label = y_indices[idx]

            for j in range(k):
                hit  = 1. * (preds[i, j] == label)
                ret += hit / k

    return ret / preds.shape[0]
