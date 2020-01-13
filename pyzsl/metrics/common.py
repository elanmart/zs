from functools import wraps
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


# utils ------------------------------------------------------------------------
@njit(parallel=False)
def _scores_to_ranks_numba(preds: np.ndarray, y_indices: np.ndarray, y_indptr: np.ndarray):
    ret  = np.zeros_like(y_indices, dtype=np.int64)

    for i in range(y_indptr.size - 1):

        (start, stop) = y_indptr[i], y_indptr[i + 1]

        inds         = y_indices[start:stop]
        all_scores   = preds[i, :]
        label_scores = all_scores[inds]

        # we need to do a bit of magic to handle ties between positive labels properly
        all_scores[inds] = -np.inf
        sorted_inds      = np.argsort(-label_scores)

        for j in range(label_scores.size):
            ret[start + j] = (all_scores >= label_scores[j]).sum()

        for j in range(sorted_inds.size):
            idx = sorted_inds[j]
            ret[start + idx] += (j + 1)

        all_scores[inds] = label_scores

    return ret


def scores_to_ranks(scores: np.ndarray, Y: sp.csr_matrix):
    data = _scores_to_ranks_numba(scores, Y.indices, Y.indptr)
    return sp.csr_matrix((data, Y.indices.copy(), Y.indptr.copy()),
                         shape=Y.shape)


def scores_to_ranks_slow(scores: np.ndarray, Y: sp.csr_matrix):
    if scores.shape[1] != Y.shape[1]:
        raise RuntimeError(
            f'Scores and Y shapes mismatch: {scores.shape[1]} != {Y.shape[1]}. :('
        )

    all_ranks  = np.argsort(-scores, axis=1).argsort(-1)
    rows, cols = Y.nonzero()

    data  = all_ranks[rows, cols] + 1
    ranks = sp.csr_matrix((data, Y.indices, Y.indptr), shape=Y.shape)

    return ranks


def scores_to_topk(scores: np.ndarray, k: int):
    if k < 50:
        retval = np.argpartition(-scores, kth=tuple(range(k)), axis=1)
    else:
        retval = np.argsort(-scores, axis=1)

    return retval[:, :k].copy()


# dispatch ---------------------------------------------------------------------
_DISPATCH_TABLE = {}
_INT_TYPES      = {np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64}
_FLOAT_TYPES    = {np.float16, np.float32, np.float64}


def _dispatch(arrtype, dtype=None):
    def decorator(f):

        if not isinstance(dtype, Iterable):
            dtypes = (dtype,)
        else:
            dtypes = dtype

        for d in dtypes:
            _DISPATCH_TABLE[(f.__name__, arrtype, d)] = f

        @wraps(f)
        def wrapper(y_true, preds, *args, **kwargs):

            arrtype_ = type(preds)
            dtype_ = preds.dtype.type

            for cls in arrtype_.__mro__:
                fn = None
                fn = fn or _DISPATCH_TABLE.get((f.__name__, cls, dtype_))
                fn = fn or _DISPATCH_TABLE.get((f.__name__, cls, None))

                if fn is not None:
                    break

            else:
                raise RuntimeError(
                    f"Cannot dispatch: arrtype={arrtype_} dtype={dtype_}")

            return fn(y_true, preds, *args, **kwargs)

        return wrapper
    return decorator


def _metric_doc(summary, accept_ints=True, accept_floats=True, accept_sparse=True):
    def decorator(f):

        inputs = ""
        indent = ' ' * 4 * 3

        if accept_ints:
            inputs += f'\n{indent}* np array if ints (see Notes!)'

        if accept_floats:
            inputs += f'\n{indent}* np array of floats  (see Notes!)'

        if accept_sparse:
            inputs += f'\n{indent}* scipy sparse matrix of ranks  (see Notes!)'

        doc = f""" {summary}

        Parameters
        ----------
        preds: Union[np.ndarray, sp.spmatrix]
            predictions of a model.
            can be either: {inputs}

        y_true: sp.csr_matrix
            Ground-truth labels. Should be a sparse, binary matrix,
            where `y_true[i, j] == 1` only if `j`-th label is assigned to`i`-th example

        Notes
        -----
        * If `preds` is an array of ints, it is assumed to contains top-k predictions for each example,
            with preds.shape[1] == k
        * If `preds` is an array of flaots, it is assumed to be an array of scores for each label and each example.
            We therefore require preds.shape == y_true.shape
        * If `preds` is a sparse matrix, it is assumed that for each example `i`, for each true label `j`
            associated with that example, `preds[i, j]` contains a rank assigned by a model to this label.
                * Ranks start at TODO(elanmart)
                * Ties should resolved in the following way: TODO(elanmart)
                * TODO(elanmart): decide whetver ranks should be
                    'number of items with higher score'
                    'number of negative items with higher score''

        Returns
        -------
        value: scalar
            computed metric value, averaged over all examples
        """

        f.__doc__ = doc
        return f
    return decorator

