# TODO(elan): change all signatures to take explicit arguments instead of Config classes
# TODO(elan): figure out if we indeed want to generate any additional fields
import array
from collections import Counter
from itertools import chain
from typing import List, Callable, Dict, Union

from pyzsl.utils.general import json_iterator
from pyzsl.utils.nlp import Tokenizer_T, Split

DEFAULT_UNK = 1


class Vocab:
    """ Vocabulary object, built to get a mapping from words in string
    format to integers.

    Parameters
    ----------
    sources :
        Dictionary mapping filenames to `field_info`.
        Each filename specified here will be used to build the vocabulary.
        We assume lines in each file are `json` objects.
        Therefore for each filename, `field_info` dict has to be specified.
        It should map item keys to transforms, that should be applied to them.
        e.g.
        sources = {'train.json': {'title': Lemmatize(), 'summary': Split()}}
        will build a vocabulary using `train.json` file, where for each line
        line['title'] will be lemmatized, and line['summary'] will be
        split as-is.
    max_size :
        Maximum size of the vocabulary. Least frequent words will be dropped
        to meet this requirement. `None` for unbounded vocab size.
    condition : fn(Dict[str, str]) -> bool
        Callable to which each item is passed.
        If returns False, item is skipped and not used when building
        vocabulary. See Notes for a comment.

    Notes
    -----
    # TODO(elanmart): figure out how to get rid of `condition` parameter.
    # it was only added because we want to skip the rows
    # in definitions.json file
    # that correspond to unseen labels.
    # But this is ugly. How else could we handle that?
    # split definitions into two files permanently?
    # split temporarly while building vocab?
    # neither of these seems ideal.
    """

    def __init__(self,
                 sources: Dict[str,
                               Dict[str, Tokenizer_T]],
                 max_size: int = None,
                 min_freq: int = 1,
                 condition: Callable[[Dict[str, str]], bool] = None):

        self._sources  = sources
        self._max_sz   = max_size
        self._min_freq = min_freq
        self._cond     = condition or (lambda item: True)

        self._counter = Counter()
        self._stoi    = None
        self._itos    = None
        self._size    = 0

        self._pad_tok   = '__pad__'
        self._merge_tok = '__merge__'
        self._unk_tok   = '__unk__'

        self._split = Split()
        self._recorded_transforms = {}

    def __len__(self):
        return self._size

    def build(self):
        """ Builds the vocabulary as specified in the __init__.

        We don't run this code in __init__ to allow for specifying a bunch of
        vocabularies beforehand, and then building only a few chosen ones.
        """

        for filename, field_info in self._sources.items():
            field_info = list(field_info.items())

            for fld, fn in field_info:
                self._recorded_transforms[fld] = fn

            for row in json_iterator(filename):

                if not self._cond(row):
                    continue

                for fld, fn in field_info:

                    data = row[fld]
                    data = fn(data)
                    self._counter.update(data)

        if self._max_sz is not None:
            self._counter = Counter({k: v
                                     for k, v in self._counter.most_common(self._max_sz)})

        if self._min_freq > 1:
            self._counter = Counter({k: v
                                     for k, v in self._counter.items()
                                     if v >= self._min_freq})

        self._stoi = self._itos = None
        self._size = len(self._counter) + 2  # + __unk__ and __merge__

        return self

    def transform_file(self, fname, field_info):
        """ Execute `self.transform_item(item, field_info)`
        for each item in `fname`
        """

        return [
            self.transform_item(item=row, field_info=field_info)
            for row in json_iterator(fname, use_tqdm=True)
        ]

    def transform_item(self,
                       item: Dict[str, str],
                       field_info: Union[List[str],
                                         Dict[str, Tokenizer_T]]) -> array.array:
        """ Transform an item `item` to integer representation.

        Item is a dict mapping field name to some text.

        If field_info is a List, it contains keys in `item` that should
        be transformed. The transformation used is the same as specified
        in the `sources` parameter of the __init__ method.

        If field_info is a dict, it maps keys in `item` to callable tokenizer
        that should be applied to them.

        Fields are concatenated using a special merge token `__merge__`.

        Returns array.array('q', ...), (long long signed int).
        """

        if isinstance(field_info, str):
            field_info = [field_info]

        if isinstance(field_info, list):
            field_info = {fld: self._recorded_transforms[fld] for fld in field_info}

        field_info = list(field_info.items())
        merge_idx  = self.get(self._merge_tok)
        ret        = []

        for idx, (fld, fn) in enumerate(field_info):
            text = item[fld]
            inds = self.to_indices(text=text, tokenizer=fn)

            if (idx != len(field_info) - 1) and (len(inds) > 0):
                inds.append(merge_idx)

            ret.extend(inds)

        ret = array.array('q', ret)
        return ret

    def to_indices(self, text: str,
                   tokenizer: Callable = None):
        """ Transofrm a string `text` transformed with `tokenizer`
        to integer representation using the built vocabulary.

        `tokenizer` defaults to `lambda text: text.split()`
        """

        tokenizer = tokenizer or self._split
        tokens    = tokenizer(text)

        return [
            self.get(token)
            for token in tokens
        ]

    def from_indices(self, indices: List[int]) -> str:
        """ Returns a string representation of a list of indices
        """

        words = [self.rev(idx)
                 for idx in indices]

        return ' '.join(words)

    def get(self, word: str):
        """ Returns index of a word `word`
        """

        return self.stoi.get(word, DEFAULT_UNK)

    def rev(self, idx: int):
        """ Returns a word corresponding to index `idx`
        """

        return self.itos[int(idx)]

    @property
    def stoi(self):
        """ str-to-int, mapping words in the vocabulary to their indices
        """

        if self._stoi is None:

            self._stoi = {}
            specials   = [self._pad_tok, self._unk_tok, self._merge_tok]
            tokens     = chain(specials, self._counter.keys())

            for name in tokens:
                self._stoi[name] = len(self._stoi)

        return self._stoi

    @property
    def itos(self):
        """ int-to-str, mapping integer indices to words in vocaulary
        """

        if self._itos is None:
            self._itos = {i: n for n, i in self.stoi.items()}
            self._itos = [self._itos[idx] for idx in range(len(self._itos))]

        return self._itos
