import numpy as np
import scipy.sparse as sp
import sklearn.metrics
from numpy import asarray as arr

from .common import _metric_doc, _dispatch, _INT_TYPES, _FLOAT_TYPES


@_metric_doc('Computes ranking loss. Equivalent to `sklearn.metrics.label_ranking_loss`', accept_ints=False)
def rankloss(y_true, preds):
    return _rankloss(y_true, preds)


@_dispatch(np.ndarray, _INT_TYPES)
def _rankloss(y_true, preds):
    raise NotImplementedError(
        "Rankloss cannot be computed given only top-k predictions! "
        ":("
    )


@_dispatch(np.ndarray, _FLOAT_TYPES)
def _rankloss(y_true: sp.csr_matrix, preds: np.ndarray):
    return sklearn.metrics.label_ranking_loss(y_true, preds)


@_dispatch(sp.csr_matrix)
def _rankloss(y_true: sp.csr_matrix, preds: sp.csr_matrix):
    n_pos = arr(preds.astype(np.bool).sum(1))
    n_neg = y_true.shape[1] - n_pos

    diff = (1 + n_pos) * (n_pos / 2)
    ret  = arr(preds.sum(1).astype(np.float32)) - diff
    ret /= (np.maximum(n_pos, 1) * np.maximum(n_neg, 1))
    ret *= (n_pos > 0)

    return ret
