import array
import hashlib
import logging
import os
import gc
import pickle
import shlex
import shutil
import subprocess
import tempfile
import ujson as json
from collections import defaultdict
from collections import namedtuple
from contextlib import contextmanager, ExitStack
from pathlib import Path
from tarfile import TarFile
from typing import Dict, List, Union, Iterable, Any, Iterator

import dill
import numpy as np
import scipy.sparse as sp
import spacy
import torch as th
from numba import njit
from numpy.lib import format as fmt
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PathBase:
    """ Convenience class which can be used to easily manage
    paths to multiple files, without specifing their names
    as strings and without using hundreds of os.path.join(...)s

    >>> class MyPaths(PathBase):
    ...     dir_1  = Path('my_dir')
    ...     file_1 = dir_1 / 'foo.txt'
    ...
    >>> paths = MyPaths('/path/to/root', as_str=True)
    >>> paths.file_1
    /path/to/root/my_dir/foo.txt
    """

    def __init__(self, root, as_str=False):
        self._root = root
        self._str  = as_str

    def _join(self, item):
        path = os.path.join(self._root, item)

        if self._str:
            return str(path)
        else:
            return Path(path)

    def __getattribute__(self, item):

        if item.startswith('_'):
            return object.__getattribute__(self, item)

        item = object.__getattribute__(self, item)

        if type(item) in {list, tuple}:
            return type(item)(self._join(v) for v in item)

        elif is_container(item):
            args = [self._join(v) for v in item]
            return type(item)(*args)

        return self._join(item)


class CsrBuilder:
    dtype_map = {
        np.bool:    'b',
        np.int32:   'i',
        np.int64:   'l',
        np.float32: 'f',
        np.float64: 'd',
    }

    def __init__(self, dtype, remove_duplicates=False):
        self.dtype = dtype
        self.dedup = remove_duplicates

        self.data    = array.array(self.dtype_map[dtype])
        self.indices = array.array('i')
        self.indptr  = array.array('l', [0])

    def update_row(self, indices, data=None, add_ones=False):

        if len(indices) == 0:
            return

        if (add_ones is True) and (data is None):
            data = [1] * len(indices)

        if data is not None:
            self.data.extend(data)

        self.indices.extend(indices)

    def add_row(self, indices, data=None):
        if len(indices) == 0:
            return

        self.update_row(indices, data)
        self.row_finished()

    def row_finished(self):
        self.indptr.append(len(self.indices))

    def get(self, n_rows=None, n_cols=None):

        if len(self.data) == 0:
            data = np.ones((len(self.indices), ), dtype=self.dtype)

        else:
            data = np.frombuffer(self.data, dtype=self.dtype)

            self.data = []
            gc.collect()

        indices = np.frombuffer(self.indices, dtype=np.int32)
        indptr  = np.frombuffer(self.indptr,  dtype=np.int64)
        n_rows  = n_rows or len(indptr) - 1
        n_cols  = n_cols or np.max(indices) + 1

        csr = sp.csr_matrix((data, indices, indptr), dtype=self.dtype, shape=(n_rows, n_cols))
        _   = csr.check_format(full_check=True)  # !!!

        if self.dedup:
            csr.sum_duplicates()
            csr.sort_indices()

        return csr


def sparse_div(X: sp.csr_matrix, v: np.ndarray):
    X = X.copy()
    return sparse_div_(X, v)


def sparse_div_(X: sp.csr_matrix, v: np.ndarray):

    if v.ndim == 1:
        assert v.shape[0] == X.shape[1], f"Shape mismatch: " \
                                         f"v.shape[0] != X.shape[1]: " \
                                         f"{v.shape[0]} != {X.shape[1]}"
        X = X.tocsc()

    if v.ndim == 2:
        assert v.shape[1] == 1, f"v is not a column vector. It has shape: {v.shape}"
        assert v.shape[0] == X.shape[0], f"Shape mismatch: " \
                                         f"v.shape[0] != X.shape[1]: " \
                                         f"{v.shape[0]} != {X.shape[0]}"
        v = v.reshape(-1)
        X = X.tocsr()

    _sparse_div_inplace_numba(X.indices, X.indptr, X.data, v)
    return X.tocsr(copy=False)


@njit
def _sparse_div_inplace_numba(indices, indptr, data, v):
    for ix in range(indptr.shape[0] - 1):

        (start,
         stop) = indptr[ix], indptr[ix + 1]
        value = v[ix]

        for i in range(start, stop):
            data[i] /= value


@contextmanager
def mktemp():
    """ Contextmanager yielding a path to a temporary,
    closed file, which is deleted on exit.
    """

    fd, fname = tempfile.mkstemp()
    os.close(fd)

    yield fname

    if os.path.exists(fname):
        os.remove(fname)


def named_tempfile():
    fd, fname = tempfile.mkstemp()
    os.close(fd)

    return fname


def is_container(x):
    return isinstance(x, Iterable) and (not isinstance(x, str))


def dinv(d: Dict):
    """ Invert dictionary, mapping values to keys

    >>> dinv({'foo': 1, 'bar': 2})
    {1: 'foo', 2: 'bar'}
    """

    assert len(set(d.values())) == len(d), "Dictionary values must contain only unique elements"
    return {val: key for key, val in d.items()}


def asdict(l: List):
    """ maps each item in a list to its position in this list.
    Makes sure list contains only unique elements.

    >>> asdict(['foo', 'bar'])
    {'foo': 0, 'bar': 1}
    """

    assert len(set(l)) == len(l), "List must contain only unique elements"
    return {v: i
            for i, v in enumerate(l)}


def aslist(d: Dict):
    return [d[idx] for idx in range(len(d))]


def maybe_tqdm_open(path, flag):
    if flag:
        return tqdm_open(path)
    else:
        return open(path)


def json_iterator(path: str, use_tqdm=True) -> Iterator[Dict[str, Any]]:
    """ Iterator yielding json.loads(line) for each line in file at `path`.
    Skips empty lines. Uses `ujson` for faster parsing.
    """

    with maybe_tqdm_open(path, flag=use_tqdm) as f:
        for line in f:

            line = line.strip()
            if len(line) == 0:
                continue

            yield json.loads(line)


def normalize_path(path: Union[str, Path]):
    """ Expand user and return real, absolute path to `path`.

    Examples
    --------
    >>> normalize_path('~/foo')
    /home/user/foo
    >>> normalize_path('./foo.txt')
    /home/user/current/directory/foo.txt
    """

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    path = os.path.realpath(path)

    return path


def as_cuda(*x: Union[th.Tensor, th.nn.Module], cuda=False):
    """ Move each item in `x` to `cuda` devide.
    Works recursively with lists and tuples.
    """

    if len(x) == 1:
        x, = x

    if x is None:
        return x

    typ = type(x)
    if typ in {list, tuple}:
        return typ(as_cuda(item, cuda=cuda) for item in x)

    if cuda:
        x = x.to('cuda', non_blocking=True)

    return x


def readlines(fname):
    """ equivalent to [line.strip() for line in open(fname)]
    """

    with open(fname) as f:
        ret = [l.strip()
               for l in f.readlines()]

    return ret


def json_load(fname: str) -> Union[Dict[str, Any], List[Any]]:
    """ Load pickle object from `fname` using json
    """

    with open(fname) as f:
        return json.load(f)


def json_dump(obj, fname):
    """ Save object `obj` to `fname` using json
    """

    with open(fname, 'w') as f:
        json.dump(obj, f, indent=4)


def pickle_dump(obj, fname):
    """ Save object `obj` to `fname` using pickle
    """

    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fname):
    """ Load pickle object from `fname` using pickle
    """

    with open(fname, 'rb') as f:
        obj = pickle.load(f)

    return obj


def dill_dump(obj, fname):
    """ Save object `obj` to `fname` using dill
    """

    with open(fname, 'wb') as f:
        dill.dump(obj, f)


def dill_load(fname):
    """ Load pickle object from `fname` using dill
    """

    with open(fname, 'rb') as f:
        obj = dill.load(f)

    return obj


def scipy_dump(obj: sp.csr_matrix, fname):
    sp.save_npz(fname, obj)


def scipy_load(fname) -> sp.csr_matrix:
    X = sp.load_npz(fname)  # type: sp.csr_matrix
    X.check_format(full_check=True)

    return X


def scipy_load_optimized(fname) -> sp.csr_matrix:
    X = sp.load_npz(fname)  # type: sp.csr_matrix
    X.sum_duplicates()
    X.sort_indices()
    X.check_format(full_check=True)

    return X


def copytree(src: str, dst: str,
             force=False, ignore_errors=True, **kwargs):
    """ Calls `shutil.copytree`, but with optional error hadnling.

    Parameters
    ----------
    src :
        source directory
    dst :
        destination directory
    force :
        if True, before copying `dst` directory is removed if it exists
    ignore_errors :
        if False, `dst` directory must either not exist, or be removed with `force`.
    kwargs :
        additional keyword args are passed to shutil.copytree
    """

    if os.path.exists(dst) and force:
        shutil.rmtree(dst)

    if os.path.exists(dst) and ignore_errors:
        return

    return shutil.copytree(src, dst, **kwargs)


def _check_spacy():
    # TODO(elan): remove this after spacy download is added to setup.py

    try:
        spacy.load('en')
    except OSError:
        subprocess.check_call('python -m spacy download en'.split())


def get_filename(path: str) -> str:
    """ Given a full path, return a filename without extension
    """

    return os.path.splitext(os.path.basename(path))[0]


def obj_name(obj):
    try:
        return obj.__name__
    except AttributeError:
        return obj.__class__.__name__


def file_md5(fname, ref):
    with open(fname, 'rb') as f:
        data_md5(f.read(), ref)


def data_md5(data, ref):
    checksum = hashlib.md5(data).hexdigest()
    logger.info(f'MD5 check: {checksum} vs {ref}')

    assert checksum == ref, f"Checksum {checksum} didn't match reference {ref}"


def extract_from_tarfile(path, member, output_path, md5=''):
    with TarFile(path) as f:
        data = f.extractfile(member).read()

    if md5 != '':
        data_md5(data, md5)

    with open(output_path, 'wb') as f:
        f.write(data)


def restricted_pickle_load(fname):
    """ https://docs.python.org/3.4/library/pickle.html#restricting-globals
    """

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            raise pickle.UnpicklingError("Access forbidden.")

    with open(fname, 'rb') as f:
        return RestrictedUnpickler(f).load()


@contextmanager
def get_contexts(managers):

    if not isinstance(managers, Iterable):
        managers = [managers]

    stack    = ExitStack()
    contexts = [stack.enter_context(m) for m in managers]

    if len(contexts) == 1:
        contexts, = contexts

    with stack:
        yield contexts


def numpy_dump(obj, fname):
    with open(fname, 'wb') as f:
        obj = np.asanyarray(obj)
        fmt.write_array(f, obj)


def numpy_load(fname: str) -> np.ndarray:
    return np.load(fname)

def numpy_array_from_text(fname: str,
                           sep=" ", dtype=np.float, **kwargs) -> np.ndarray:
    """ Reads input from text file and reshapes:
            into 1D array if cols or row == 1
            into matrix otherwise
        It assumes one line = one row
    """
    matrix = np.fromfile(fname, sep=sep, dtype=dtype, **kwargs)
    with open(fname) as f:
        line = f.readline()
    cols = len(line.split())
    assert matrix.size % cols == 0
    rows = matrix.size // cols
    if rows == 1 or cols == 1:
        return matrix
    else:
        return matrix.reshape(rows, cols)


def indexer():
    dct = defaultdict()
    dct.default_factory = dct.__len__

    return dct


def as_named_tuple(name, **kwargs):
    kwargs = tuple(kwargs.items())
    (fields,
     values) = zip(*kwargs)

    T = namedtuple(name, fields)
    return T(*values)


@contextmanager
def tqdm_open(filename):
    total = os.path.getsize(filename)
    pb    = tqdm(total=total, unit="B", unit_scale=True,
                 desc=os.path.basename(filename), miniters=1)

    def wrapped_line_iterator(fd):
        processed_bytes = 0
        for line in fd:
            processed_bytes += len(line)
            # update progress every KB.
            if processed_bytes >= 1024:
                pb.update(processed_bytes)
                processed_bytes = 0

            yield line

        # finally
        pb.update(processed_bytes)
        pb.close()

    with open(filename) as fd:
        yield wrapped_line_iterator(fd)


def dprint(msg):
    print('=' * 100)
    print(msg)
    print('=' * 100)


def vec(x, dtype=None):
    dtype = dtype or x.dtype

    if sp.issparse(x):
        x = x.todense()

    return np.asarray(x).reshape(-1).astype(dtype)


class CommandRunner:
    def __init__(self, cmd):
        self.cmd    = cmd
        self.proc   = None
        self.stdout = b''
        self.stderr = b''
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, raise_errors=False):
        self.logger.debug(f'Running command: {self.cmd}')

        self.proc = subprocess.Popen(
            shlex.split(self.cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        (self.stdout,
         self.stderr) = [pipe.decode()
                         for pipe in self.proc.communicate()]

        if (not self.ok()) and raise_errors:
            self.raise_error()

        return self

    def ok(self):
        return self.proc.returncode == 0

    def get_boxed_output(self, stdout=False, stderr=True):
        dash = '=' * 100 + '\n'
        msg  = f'{dash}'

        if stdout:
            msg += f'STDOUT: {self.stdout}\n' \
                   f'{dash}'

        if stderr:
            msg += f'STDERR: {self.stderr}\n' \
                   f'{dash}'

        return msg

    def raise_error(self):
        if not self.ok():
            raise subprocess.CalledProcessError(
                self.proc.returncode,
                self.cmd,
                self.stdout,
                self.stderr
            )


def load_zsl_split(path, as_set=True):
    ZslSplit = namedtuple('ZslSplit', ['seen', 'unseen'])
    obj      = json_load(path)

    if as_set:
        (obj['seen'],
         obj['unseen']) = (set(obj['seen']),
                           set(obj['unseen']))

    ret = ZslSplit(
        obj['seen'],
        obj['unseen']
    )

    return ret


def load(fname, optimize=False):
    ext = str(fname).split('.')[-1]
    fn  = {
        'npy':  numpy_load,
        'json': json_load,
        'pkl':  pickle_load,
        'dill': dill_load,
        'npz':  scipy_load_optimized if optimize else scipy_load,
    }[ext]

    return fn(fname)


class ChainedParams:
    def __init__(self, *modules, deduplicate=False):

        self._params = []

        seen = set()
        for m in modules:
            for p in m.parameters():

                if p not in seen:
                    self._params.append(p)

                if deduplicate:
                    seen.add(p)

    def __iter__(self):
        yield from self._params

    def size(self):
        return sum(p.data.numel() for p in self)


# class ETAEstimator:
#     def __init__(self):
#         raise NotImplementedError(':(')
#
#     def _(self):
#         since_last  = tm - self.last
#         since_start = tm - self.start
#         self.last   = tm
#
#         rate = self.cbk_freq / since_last
#         left = n_iters - i
#         t_left = ( left / rate)
#
#         progress = iteration / max_iterations
#
#         sec     = t_left
#         th      = t_left // 3600
#         tmin    = (t_left // 60) % 60
#         elapsed = since_start / 60
#
#         estimated_total = elapsed / progress
#         estimated_left  = estimated_total * (1 - progress)
#
#         msg = "\r {} ::: loss: {:.3f} ::: {:.3f}%, elapsed: {:.1f} [min], total: {:.1f} [min], ETA: {:.1f} [min]    "
#         msg = msg.format(td, loss, 100*progress, elapsed, estimated_total, estimated_left)
#
#         print(msg, end=self.str_end, flush=True)
