import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional
from typing import Union

import numpy as np
import torch as th
from numba import njit
from torch import no_grad
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# either a loaded array, or a path from which to read the array
LazyArray_T = Union[np.ndarray, str, Path]


def _maybe_from_numpy(item: Union[th.Tensor, np.ndarray]):
    if isinstance(item, np.ndarray):
        item = th.from_numpy(item)

    return item


def _maybe_load(item: LazyArray_T) -> np.ndarray:
    """ Item is either a loaded array, or a path from which to read the array

    """
    if not isinstance(item, np.ndarray):
        item = np.load(item)

    return item


@njit
def _get_valid_desc_idx(D, row_idx, patience) -> int:
        for _ in range(patience):
            j = random.randint(0, 9)

            if D[row_idx, j, 0] != 0:
                return j

        raise RuntimeError("Number of trials exceeded!")


@njit
def _get_negative_idx(Y, n_rows, pos_idx, patience) -> Tuple[int, int]:
        for _ in range(patience):

            neg_idx = random.randint(0, n_rows)

            if Y[neg_idx] != Y[pos_idx]:
                return neg_idx

        raise RuntimeError("Number of trials limit exceeded")


def _make_desc(D, vocab_size, max_len) -> np.ndarray:
    """ Transform index array to one-hot encoded array """

    # move to pytorch because idk how you scatter_ in numpy.
    D = (th.from_numpy(D)
           .contiguous()
           .unsqueeze(2))

    retval = th.zeros(D.size(0),   # N: num of descriptions
                      10,          # M: num descriptions per image
                      vocab_size,  # V: vocab size
                      max_len)     # L: max_len

    # TODO(elanmart): describe the magic happening here
    retval.scatter_(2, D, 1.)
    retval[:, :, 0, :] = 0  # zero-out the padding idx
    retval = retval.permute(0, 1, 3, 2)  # (N, M, L, V)

    return retval.numpy()


def _make_desc_legacy(D, idx, j, max_len, as_indices, vocab_size=None):
    """ Legacy code to create a one-hot encoding given a single description

    """

    if as_indices:
        d      = th.zeros(max_len, vocab_size, dtype=th.float32)
        tokens = D[idx, j, :].view(-1, 1)

        d.scatter_(1, tokens, 1.)
        d[:, 0] = 0  # zero-out the padding idx

    else:
        d = D[idx, j, :]

    return d


class CubDataset(Dataset):
    def __init__(
            self,
            X: LazyArray_T,
            Y: LazyArray_T,
            D: LazyArray_T,
            IDs: LazyArray_T,
            R: Optional[np.ndarray] = None,
            return_indices=False,
            max_len=201,  # from the paper
            swap_image=False,
            return_negative=False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.X = _maybe_load(X)
        self.Y = _maybe_load(Y)
        self.D = _maybe_load(D)
        self.I = _maybe_load(IDs)
        self.R = R

        self.ret_indices = return_indices
        self.max_len     = max_len
        self.swap_image  = swap_image
        self.negative    = return_negative
        self.n_classes   = len({y for y in Y})
        self.vocab_size  = D.max() + 1
        self._patience   = 100

        self._y2indx = defaultdict(list)
        for idx, y in enumerate(self.Y):
            self._y2indx[y].append(idx)

        if self.R is None:

            if self.ret_indices:
                self.R = _make_desc(D, vocab_size=self.vocab_size, max_len=self.max_len)
            else:
                self.R = self.D

        self.logger.info(f'Shape of self.X = {self.X.shape}')
        self.logger.info(f'Shape of self.Y = {self.Y.shape}')
        self.logger.info(f'Shape of self.D = {self.D.shape}')
        self.logger.info(f'Shape of self.R = {self.R.shape}')

    def __len__(self):
        return self.X.shape[0]

    def _get_valid_desc_idx(self, row_idx) -> int:
        return _get_valid_desc_idx(self.D, row_idx, self._patience)

    def _get_negative_idx(self, pos_idx) -> Tuple[int, int]:
        n_rows = self.Y.size(0) - 1
        return _get_negative_idx(self.Y, n_rows, pos_idx, self._patience)

    def _make_single_desc(self, idx, j):
        return _make_desc_legacy(self.D, idx, j, self.max_len, self.ret_indices, self.vocab_size)

    def __getitem__(self, index: int):

        y_pos = self.Y[index, ...]

        # TODO(elanmart): document what's going on in here.
        if self.swap_image:
            swapped_idx = random.choice(self._y2indx[y_pos])
            x = self.X[swapped_idx, ...]
        else:
            x = self.X[index, ...]

        d_idx = self.I[index]
        j     = self._get_valid_desc_idx(d_idx)
        d_pos = self.R[d_idx, j, ...]  # TODO(elanmart): using this may give OOM
        # d_pos = self._make_single_desc(index, j)

        if self.negative:
            neg_index = self._get_negative_idx(index)
            neg_d_idx = self.I[neg_index]
            neg_j = self._get_valid_desc_idx(neg_d_idx)
            d_neg = self.R[neg_d_idx, neg_j, ...]  # TODO(elanmart): using this gives OOM
            # d_neg  = self._make_single_desc(neg_idx, neg_j)

            retval = x, d_pos, d_neg

        else:
            retval = x, d_pos, y_pos

        return [
            _maybe_from_numpy(item)
            for item in retval
        ]


class TransferDataset(Dataset):
    def __init__(self,
                 src: CubDataset,
                 dst: CubDataset,
                 size: int):

        self.src = src
        self.dst = dst
        self._size = size

    def __len__(self):
        return self._size

    # noinspection PyCallingNonCallable
    def __getitem__(self, item):

        if random.random() > 0.5:
            data = self.src
            y    = th.tensor(1, dtype=th.float32)

        else:
            data = self.dst
            y    = th.tensor(0, dtype=th.float32)

        i       = random.randint(0, len(data) - 1)
        x, d, _ = data[i]

        return x, d, y


class TransferLoader:
    def __init__(self, *args, **kwargs):
        self._loader = DataLoader(*args, **kwargs)
        self._iter   = iter(self._loader)

    def reset(self):
        self._iter   = iter(self._loader)

    def __next__(self):
        return next(self._iter)


class CubBatchSampler:
    def __init__(self, dataset: CubDataset, batch_size: int):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self._class2inds = defaultdict(list)

        for idx, y in enumerate(self.dataset.Y):
            self._class2inds[y].append(idx)

        self._class2inds = dict(self._class2inds)
        self._classes    = list(self._class2inds.keys())
        self._size       = (len(self.dataset) // self.batch_size) + 1

    def __iter__(self):
        for _ in range(self._size):

            # choose which classes we will sample.
            # we do this because we want every example from a different class
            # this is required by the symmetric loss
            class_inds = th.randperm(len(self._classes))[:self.batch_size]
            classes    = [self._classes[idx] for idx in class_inds]

            # assert uniqnuess
            assert len(classes) == len(set(classes))

            batch = [random.choice(self._class2inds[y])  # class2idx[y] is a list of examples for this y
                     for y in classes]

            yield batch

    def __len__(self):
        return self._size


class Descriptions:
    def __init__(self,
                 full_D: np.ndarray,
                 Y: np.ndarray,
                 IDs: np.ndarray,
                 as_indices: bool,
                 max_len: int,
                 vocab_size=None):

        classes   = {int(y) for y in Y}
        n_classes = len(classes)
        I, J, K   = full_D.shape
        K         = max(K, max_len)

        # store
        self.as_indices = as_indices
        self.max_len    = max_len
        self.vocab_size = vocab_size

        # Map internal class idx to original
        self._i2c = np.array(list(classes))

        # Map class indexes to unique, contiguous integers
        self._c2i = {y: i for i, y in enumerate(self._i2c)}

        # Prepare to create the subset of descr we will actually use
        self.Y = Y
        self.D = np.zeros((n_classes, J, K), dtype=full_D.dtype)

        # we copy only the descriptions for the labels that actually occur in Y
        for i in range(self.D.shape[0]):
            c = self._i2c[i]

            # find first example of this class
            example_idx = np.nonzero(Y == c)[0][0]
            
            # full_D index includes training examples, so we index it indirectly
            d_idx = IDs[example_idx]

            d = full_D[d_idx, :, :K]

            self.D[i, :, :] = d

        # perhaps map D to sparse repr
        if self.as_indices:
            self.R = _make_desc(self.D, self.vocab_size, self.max_len)
        else:
            self.R = self.D

        self.Y = th.from_numpy(self.Y)
        self.R = th.from_numpy(self.R)

        self.D = None  # ?
        self._i2c = th.from_numpy(self._i2c)

    def to(self, *args, **kwargs):
        self.R = self.R.to(*args, **kwargs)
        self.Y = self.Y.to(*args, **kwargs)
        self._i2c = self._i2c.to(*args, **kwargs)

        return self

    def compute_representations(self,
                                model: th.nn.Module,
                                device: th.device,
                                normalize: bool = False):

        retval = []

        with no_grad():
            for i in range(self.R.size(0)):
                row  = self.R[i, ...]\
                        .contiguous()\
                        .to(device, non_blocking=True)

                repr = model(row)\
                    .mean(0)\
                    .detach()\
                    .squeeze()

                retval.append(repr)

            retval = th.stack(retval)

            if normalize:
                retval_n = F.normalize(retval).detach()
            else:
                retval_n = None

        return retval, retval_n

    def predict(self,
                scores: th.Tensor,
                mu: th.Tensor = None,
                sigma: th.Tensor = None):

        with th.no_grad():

            if mu is not None:
                scores = (scores - mu) / sigma

            _, inds = scores.max(1)
            return self._i2c[inds]


class GanDataset(Dataset):
    """ TODO: This was NOT reviewed after recent (25.08.2018) small refactor

    """

    def __init__(self, src):
        raise NotImplementedError("Please review this code before using it.")

        self.X = src.X
        self.Y = src.Y
        self.D = src.repr

    def __getitem__(self, index):
        return self.X[index], self.D[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)
