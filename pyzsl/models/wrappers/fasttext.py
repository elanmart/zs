import array

import numpy
from fastText import FastText as _FastTextLib
from scipy.sparse import csr_matrix

from pyzsl.utils.general import maybe_tqdm_open, dinv, aslist, CsrBuilder
from .base import BaseWrapper


# noinspection PyPep8Naming
class FastText(BaseWrapper):
    def __init__(
            self,
            name_to_index,
            lr                = 0.01,
            dim               = 256,
            epoch             = 50,
            wordNgrams        = 1,
            loss              = "hs",
            neg               = 5,
            bucket            = 2000000,
            thread            = 8,
            lrUpdateRate      = 100,
            t                 = 1e-4,
            label             = "__label__",
            verbose           = 0,
            tqdm              = False,
    ):
        super().__init__()

        if name_to_index is not None:
            name_to_index = {
                label + k.replace(' ', '_'): v
                for k, v in name_to_index.items()
            }

        self.params = dict(
            lr=lr, dim=dim, epoch=epoch, wordNgrams=wordNgrams, loss=loss,
            neg=neg, bucket=bucket, thread=thread, lrUpdateRate=lrUpdateRate,
            t=t, label=label, verbose=verbose,
        )

        self.tqdm   = tqdm
        self.model_ = None
        self.stoi_  = name_to_index
        self.itos_  = None

    def fit(self, path):
        self.model_ = _FastTextLib.train_supervised(path, **self.params)

        if self.stoi_ is None:
            itos       = self.model_.get_labels()
            self.stoi_ = {name: idx for idx, name in enumerate(itos)}

        self.itos_ = aslist(dinv(self.stoi_))

        return self

    def _predict_lines(self, path, k):
        with maybe_tqdm_open(path, flag=self.tqdm) as f:
            for line in f:

                line          = line.rstrip()
                names, scores = self.model_.predict(line, k=k)
                indices       = [self.stoi_[n] for n in names]

                yield (numpy.array(indices),
                       scores)

    def predict_topk(self, path, *, k, tqdm=False):
        ret = array.array('l')

        for indices, _ in self._predict_lines(path, k):
            ret.extend(indices)

        arr = (numpy.frombuffer(ret, dtype=numpy.int64)
                    .reshape(-1, k))

        return arr

    def predict_scores(self, path):
        m   = len(self.stoi_)
        ret = []

        for indices, scores in self._predict_lines(path, m):

            row = numpy.zeros((m, ))
            row[indices] = scores

            ret.append(row)

        return numpy.stack(ret)

    def predict_ranks(self, path: str, Y: csr_matrix):

        builder = CsrBuilder(dtype=numpy.int32)
        m       = len(self.stoi_)

        for row_y, row_pred in zip(Y, self._predict_lines(path, m)):
            row_y   = row_y.indices
            inds, _ = row_pred

            idx_to_rank = {idx: rank for rank, idx in enumerate(inds)}
            ranks = [idx_to_rank.get(y, m) for y in row_y]

            builder.add_row(indices=row_y, data=ranks)

        return builder.get()

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'{self.__class__.__name__}({params})'
