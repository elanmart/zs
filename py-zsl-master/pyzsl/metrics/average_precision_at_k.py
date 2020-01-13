import numpy as np
import scipy.sparse as sp
from numba import njit

from .common import _metric_doc, _dispatch, scores_to_topk, _INT_TYPES, \
    _FLOAT_TYPES, scores_to_ranks


@_metric_doc('Computes AP@k, see http://fastml.com/what-you-wanted-to-know-about-mean-average-precision/')
def average_precision_at_k(y_true, preds, k=1):
    return _average_precision_at_k(y_true, preds, k)


@_dispatch(np.ndarray, _INT_TYPES)
def _average_precision_at_k(y_true: sp.csr_matrix, preds: np.ndarray, k: int):
    return _average_precision_at_k_topk(y_indices=y_true.indices, y_indptr=y_true.indptr, topk=preds)


@_dispatch(np.ndarray, _FLOAT_TYPES)
def _average_precision_at_k(y_true: sp.csr_matrix, preds: np.ndarray, k: int):
    ranks = scores_to_ranks(scores=preds, Y=y_true)
    return _average_precision_at_k_ranks(ranks_data=ranks.data,
                                         ranks_indices=ranks.indices,
                                         ranks_indptr=ranks.indptr,
                                         k=k)


@_dispatch(sp.csr_matrix)
def _average_precision_at_k(y_true: sp.csr_matrix, preds: sp.csr_matrix, k: int):
    return _average_precision_at_k_ranks(ranks_data=preds.data,
                                         ranks_indices=preds.indices, ranks_indptr=preds.indptr, k=k)


@njit
def _average_precision_at_k_topk(y_indices, y_indptr,  # y in sparse format
                                 topk):
    retval = 0.0

    for i, (start, stop) in enumerate(zip(y_indptr, y_indptr[1:])):

        labels = set(y_indices[start:stop])
        preds  = topk[i, :]
        div    = min(len(labels), preds.size)

        hits = 0.
        AP   = 0.

        for j, p in enumerate(preds, 1):

            if p in labels:
                hits += 1.
                AP   += (hits / j) * (1 / div)  # precision-at-i * change-in-recall-at-1

        retval += AP / topk.shape[0]

    return retval


@njit
def _average_precision_at_k_ranks(ranks_data, ranks_indices, ranks_indptr, k):

    retval = 0.0

    for i, (start, stop) in enumerate(zip(ranks_indptr, ranks_indptr[1:])):

        ranks = ranks_data[start:stop]
        ranks = ranks[ranks <= k]

        if ranks.size == 0:
            continue

        ranks.sort()
        hits = 0.
        AP   = 0.
        div  = min(stop - start, k)
        n    = ranks_indptr.shape[0] - 1

        for r in ranks:
            hits += 1.
            AP   += (hits / r) * (1 / div)  # precision-at-i * change-in-recall-at-1
        retval += AP / n

    return retval
