import gc
import logging

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from pyzsl.metrics import scores_to_topk, scores_to_ranks
from pyzsl.models.wrappers.base import BaseWrapper


class SimilarityModel(BaseWrapper):
    def __init__(self, metric='cosine', batch_size=4096, n_jobs=-1):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.metric     = metric
        self.batch_size = batch_size
        self.n_jobs     = n_jobs

        # we accept only sklearn metrics
        assert metric in {'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'}

    def _run(self, X, D):

        N = X.shape[0]
        I = np.array(list(
            range(0, N, self.batch_size)
        ))

        self._logger.debug('Entering inner loop')
        for idx in tqdm(I):
            gc.collect()

            (start,
             stop)  = idx, min(idx + self.batch_size, N)

            subset    = X[start:stop, :]
            distances = pairwise_distances(subset, D, metric=self.metric, n_jobs=self.n_jobs)
            yield (
                (start, stop),
                -1 * distances
            )

    def predict_topk(self,
                     X: sp.csr_matrix,
                     D: sp.csr_matrix,
                     k: int = 1):

        self._logger.debug('Running predict-topk')
        retval = np.zeros((X.shape[0], k), dtype=np.int64)

        self._logger.debug('return buffer allocated, running the loop')
        for (ix_start, ix_stop), scores in self._run(X, D):
            topk = scores_to_topk(scores, k=k)
            retval[ix_start:ix_stop, :] = topk

        self._logger.debug('All done.')
        return retval

    def predict_ranks(self,
                      X: sp.csr_matrix,
                      D: sp.csr_matrix,
                      Y: sp.csr_matrix):

        self._logger.debug('Running predict-ranks')
        retval = sp.csr_matrix((0, Y.shape[1]), dtype=np.int32)

        self._logger.debug('return buffer allocated, running the loop')
        for (ix_start, ix_stop), scores in self._run(X, D):
            y_rows = Y[ix_start:ix_stop, :]
            ranks  = scores_to_ranks(scores, y_rows)

            retval = sp.vstack((retval, ranks))

            del scores
            gc.collect()

        self._logger.debug('All done.')
        return retval

    def predict_scores(self,
                       X: sp.csr_matrix,
                       D: sp.csr_matrix):

        retval = np.zeros((X.shape[0], D.shape[0]), dtype=np.float32)

        for (ix_start, ix_stop), scores in self._run(X, D):
            retval[ix_start:ix_stop, :] = scores

        return retval
