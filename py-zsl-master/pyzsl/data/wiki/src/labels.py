import logging
import ujson as json
import warnings
from collections import defaultdict
from itertools import chain
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from pyzsl.data.wiki.paths import DataStub
from pyzsl.data.wiki.src.dict_keys import DefinitionKeys, EntryKeys
from pyzsl.utils.general import json_iterator, indexer, dinv, json_load, \
    numpy_load, load_zsl_split

try:
    from numba import njit
except ImportError:
    warnings.warn('Numba is not installed. Some functions '
                  'may run slower due to numba.jit being unavaiable.')
    njit = lambda f: f


def make_label_split(doc_path: Tuple[str, str, str],
                     def_path: str,
                     min_freq: int,
                     output_path: str):
    """ Split the categories found in the dataset into "seen" / "unseen"
    groups for zero-shot learning.

    Write the result as a json object with two keys
    ({"seen": ["cat-1", "cat-2", ...], "unseen": [...]})
    into a file at `output_path`.

    Parameters
    ----------
    doc_path :
        Paths to train, development, and test data (documents)
    def_path :
        Path to a file with definitions of each category
    min_freq :
        Minimum frequency in the train set of a category
        to consider it "seen". Labels with lower frequency are marked as
        "unseen"
    output_path :
        path where the output will be written to.
    """

    _logger = logging.getLogger(make_label_split.__name__)

    _all    = set()
    _seen   = set()
    _unseen = set()
    _count  = defaultdict(int)

    (_train,
     _dev,
     _test) = doc_path
    _def = def_path
    _min_fr = min_freq

    K_df = DefinitionKeys
    K_en = EntryKeys

    def _log(prefix):
        _logger.debug(prefix + f'|seen| = {len(_seen)}, |unseen| = {len(_unseen)}, '
                               f'|all| = {len(_all)}')

    # get all labels from a definitions file
    for item in json_iterator(_def):
        _all.add(item[K_df.name])

    # get seen labels and count them
    for item in json_iterator(_train):
        for c in item[K_en.categories]:
            _count[c] += 1
            _seen.add(c)

    # get labels that did not appear in the trainset
    for item in chain(json_iterator(_dev), json_iterator(_test)):
        _unseen |= {c for c in item[K_en.categories] if c not in _seen}

    _log('Stats before splitting: ')

    # move the rare labels to unseen
    for k, v in _count.items():
        if v < _min_fr:
            _seen.remove(k)
            _unseen.add(k)

    _log('Stats after splitting: ')

    with open(output_path, 'w') as f:
        json.dump({
            'seen': _seen,
            'unseen': _unseen
        }, f)


def rearrange_definitions(definitions_path, seen, unseen):
    K = DefinitionKeys

    seen_rows   = []
    unseen_rows = []

    for line in json_iterator(definitions_path):
        if line[K.name] in seen:
            seen_rows.append(line)
        else:
            unseen_rows.append(line)

    with open(definitions_path, 'w') as f:
        for line in seen_rows:
            print(json.dumps(line), file=f)

        for line in unseen_rows:
            print(json.dumps(line), file=f)


def get_seen_unseen_from_data(json_paths: DataStub):

    seen = set()
    all_ = set()
    K    = EntryKeys

    def _update(path, set_):
        for row in json_iterator(path):
            for c in row[K.categories]:
                set_.add(c)

    _update(json_paths.train, seen)
    _update(json_paths.dev,   all_)
    _update(json_paths.test,  all_)

    unseen = (all_ - seen)

    return seen, unseen


def make_label_vocabs(json_paths: DataStub):

    (seen,
     unseen) = get_seen_unseen_from_data(json_paths)

    rearrange_definitions(json_paths.definitions, seen=seen, unseen=unseen)

    k       = DefinitionKeys
    stoi    = indexer()
    labels  = []

    for line in json_iterator(json_paths.definitions):
        name = line[k.name]
        label_idx = stoi[name]
        _         = labels.append(label_idx)

    stoi   = dict(stoi)
    itos   = dinv(stoi)
    itos   = [itos[idx] for idx in range(len(itos))]
    labels = np.array(labels)

    nnz, = np.diff(labels).nonzero()
    nnz  = [0] + list(nnz + 1) + [len(labels)]

    intervals = [(start, stop) for start, stop in zip(nnz, nnz[1:])]
    intervals = np.array(intervals)

    zsl_split = {'seen': seen, 'unseen': unseen}

    return stoi, itos, labels, intervals, zsl_split


def filter_unused(seen_unseen_path: str, y_stoi_path: str):

    seen_unseen = json_load(seen_unseen_path)
    y_stoi      = json_load(y_stoi_path)

    _s = 'seen'
    _u = 'unseen'

    dct = {
        _s: [name for name in seen_unseen[_s] if name in y_stoi],
        _u: [name for name in seen_unseen[_u] if name in y_stoi],
    }

    seen   = np.array([y_stoi[name] for name in dct[_s]])
    unseen = np.array([y_stoi[name] for name in dct[_u]])

    return dct, seen, unseen


def seen_definitions_mask(indices_path, intervals_path):
    indices   = numpy_load(indices_path)['seen']
    intervals = numpy_load(intervals_path)
    output    = np.zeros((intervals[-1, -1], ), dtype=np.bool)

    @njit
    def unfold_(M, I, out):

        for idx in M:
            start, stop = I[idx]
            for i in range(start, stop):
                out[i] = True

        return out

    return unfold_(indices, intervals, output)


def get_label_matrix(docs: str, y_stoi: str):
    y_stoi = json_load(y_stoi)

    rows = []
    cols = []
    n, m = 0, len(y_stoi)

    for i, row in enumerate(json_iterator(docs)):
        categories = row['categories']
        indices = [y_stoi[w] for w in categories]

        rows.extend([i] * len(indices))
        cols.extend(indices)

        n += 1

    d = np.ones((len(rows),), dtype=np.int64)
    y = sp.csr_matrix((d, (rows, cols)), shape=(n, m))

    return y


class LabelSplitter:
    """ Used to stream items to either of two files. If item does not contain
    any categories marked as "unseen", it is moved to the first file. Else,
    it should be written to the second one.

    Parameters
    ----------
    split: str
        Path to a file generated with make_label_split(...) call.

    """

    def __init__(self, split):
        obj = load_zsl_split(split)

        self._seen   = obj.seen
        self._unseen = obj.unseen

    def __call__(self, item):
        K = EntryKeys

        if any(c in self._unseen for c in item[K.categories]):
            return None, item
        return item, None
