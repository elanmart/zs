import numpy as np
import scipy.sparse as sp

from pyzsl.utils.general import vec
from .common import _metric_doc, _dispatch, _INT_TYPES, _FLOAT_TYPES


@_metric_doc('Computes reciprocal rank. See en.wikipedia.org/wiki/Mean_reciprocal_rank', accept_ints=False)
def reciprocal_rank(y_true, preds):
    return _reciprocal_rank(y_true, preds)


@_dispatch(np.ndarray, _INT_TYPES)
def _reciprocal_rank(y_true, preds):
    raise NotImplementedError(
        "reciprocal rank can not be implemented for top-k predictions :("
    )


@_dispatch(np.ndarray, _FLOAT_TYPES)
def _reciprocal_rank(y_true, preds):
    eps    = 1e-6
    y_copy = y_true.copy()
    (rows,
     cols) = y_copy.nonzero()

    y_copy.data = preds[rows, cols]

    max_value    = (y_copy.data.max() + eps)
    y_copy.data -= max_value

    y_min_relevant   = y_copy.max(axis=1).toarray() + max_value - eps
    y_mask           = vec(y_true.sum(1)) == 0
    coverage         = (preds >= y_min_relevant).sum(axis=1)
    coverage[y_mask] = 0.

    return np.mean(coverage)


@_dispatch(sp.csr_matrix)
def _reciprocal_rank(y_true: sp.csr_matrix,
                     preds: sp.csr_matrix):

    preds      = preds.copy()
    preds.data = 1. / preds.data
    inverse    = vec(preds.max(axis=1))
    retval     = 1. / inverse

    return np.mean(retval)
