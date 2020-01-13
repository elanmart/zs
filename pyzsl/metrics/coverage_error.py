from functools import wraps
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import sklearn.metrics
from numba import njit
from numpy import asarray as arr

from pyzsl.utils.general import vec

from .common import _metric_doc, _dispatch, scores_to_ranks, scores_to_topk, _INT_TYPES, _FLOAT_TYPES


@_metric_doc('Computes coverage error. Equivalent to `sklearn.metrics.coverage_error`', accept_ints=False)
def coverage_error(y_true, preds):

    if y_true.data.size == 0:
        return 0.

    return _coverage_error(y_true, preds)


@_dispatch(np.ndarray, _INT_TYPES)
def _coverage_error(y_true, preds):
    raise NotImplementedError(
        "Coverage error cannot be computed given only top-k predictions! "
        ":("
    )


@_dispatch(np.ndarray, _FLOAT_TYPES)
def _coverage_error(y_true: sp.csr_matrix, preds: np.ndarray):
    eps    = 1e-6
    y_copy = y_true.copy()
    (rows,
     cols)  = y_copy.nonzero()

    y_copy.data = preds[rows, cols]

    max_value    = (y_copy.data.max() + eps)
    y_copy.data -= max_value

    y_min_relevant   = y_copy.min(axis=1).toarray() + max_value - eps
    y_mask           = vec(y_true.sum(1)) == 0
    coverage         = (preds >= y_min_relevant).sum(axis=1)
    coverage[y_mask] = 0.

    return np.mean(coverage)


@_dispatch(sp.csr_matrix)
def _coverage_error(y_true: sp.csr_matrix, preds: sp.csr_matrix):
    return np.mean(vec(
        preds.max(axis=1)
    ))
