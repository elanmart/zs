# TODO(elanmart): rename functions to get rid of the underscores

import os
import re
import subprocess
import tempfile
from typing import List, Union, TypeVar, Optional, Callable

import spacy
from spacy.tokens import Doc, Token, Span
from spacy.lang.en import English
from nltk import WordNetLemmatizer

from pyzsl.utils.general import dprint

_SPACY_CACHE  = {}
_TOKENS_CACHE = {}

Tokenizer_T = Callable[[str], List[str]]
DocItem_T   = Union[Token, Span]
StrList_O   = Optional[List[str]]
T           = TypeVar('T')


def get_spacy(s: str) -> English:
    """ Makes sure the spacy model is not loaded multiple times
    by caching the result of spacy.load(s).
    """

    if s not in _SPACY_CACHE:
        _SPACY_CACHE[s] = spacy.load(s)

    return _SPACY_CACHE[s]


def get_spacy_small(s: str, disable_tagger=False) -> English:
    if s not in _TOKENS_CACHE:
        _TOKENS_CACHE[s] = spacy.load(s)
        _TOKENS_CACHE[s].disable_pipes('parser', 'ner')

        if disable_tagger:
            _TOKENS_CACHE[s].disable_pipes('tagger')

    return _TOKENS_CACHE[s]


def nlp_en():
    """ return spacy.load('en')
    """

    return get_spacy('en')


def nlp_large():
    """ return spacy.load('en_core_web_lg')
    """

    return get_spacy('en_core_web_lg')


def vec_large():
    """ return spacy.load('en_vectors_web_lg')
    """

    return get_spacy('en_vectors_web_lg')


def shuf(f_in: str,
         f_out: str) -> None:
    """ Shuffle the contents of file f_in and write them to f_out.

    Uses shell function `shuf` via `subprocess.check_call`
    """

    cmd = f'shuf {f_in} -o {f_out}'.split()
    _   = subprocess.check_call(cmd)


def shuf_(f_in: str) -> None:
    """ Shuffles file `f_in` inplace (by writing shuffled contnet to a temporary file)

    Uses shell function `shuf` via `subprocess.check_call`
    """

    fd, fname = tempfile.mkstemp(text=True)
    os.close(fd)

    try:
        shuf(f_in, fname)
        os.rename(fname, f_in)

    finally:
        if os.path.exists(fname):
            os.remove(fname)


def append_(sources: List[str],
            dest: str) -> None:
    """ Appends content of each file in `sources` to a file `dest`
    """

    with open(dest, 'a') as f_out:
        for src in sources:
            with open(src) as f_in:

                for line in f_in:
                    f_out.write(line)


def n_words(t):
    """ Returns number of tokens in text `t`
    """

    return len(t.split())


def truncate(t, n):
    """ Truncate text `t` to maximum of `n` words
    """

    t = t.split()
    t = t[:n]
    t = ' '.join(t)

    return t


def is_stop(t: DocItem_T):
    """ Returns True if `t` is a spacy `Token` and is a stopword
    """

    if isinstance(t, Token):
        return t.is_stop
    else:
        return False


def is_span(t: DocItem_T) -> bool:
    """ Returns True if `t` is of type spacy.Span
    """

    return isinstance(t, Span)


def _is_date(item: DocItem_T) -> bool:
    if is_span(item):
        has_num = re.match('.*[0-9]+', item.text) is not None
        has_dtype = item.label_ in {'CARDINAL', 'DATE'}

        return has_dtype and has_num

    return False


def to_list(s: Union[T, List[T]]) -> List[T]:
    """ If is not a list, return a one-elementh list [s]
    """

    if not isinstance(s, list):
        s = [s]

    return s


def to_chunks(d: str,
              max_len: int) -> List[str]:
    """ Given string `d`, split it into chunks, where each chunk
    is of not longer than `max_len`.
    """

    ret  = []
    d    = d.split()
    inds = [i for i in range(0, len(d), max_len)]

    for low_idx, high_idx in zip(inds, inds[1:]):
        item = d[low_idx:high_idx]
        item = ' '.join(item)
        ret.append(item)

    return ret


def parse(doc: Doc) -> List[DocItem_T]:
    """ parse "buildings of Los Angeles" -> ["buildings", "of", "Los Angeles"]
    """

    ents     = iter(doc.ents)
    not_ner  = {'O', ''}
    skip_ner = 'I'
    tokens   = [
        token if (token.ent_iob_ in not_ner) else next(ents)
        for token in doc
        if token.ent_iob_ != skip_ner
    ]

    return tokens


class Tokenizer:
    """ Usefull to tokenize / lowercase / lemmatize a piece of text
    Returns a list of tokens.
    """

    def __init__(self, tokenize=False, lowercase=False, lemmatize=False):

        self._active = any([tokenize, lowercase, lemmatize])

        self._lemmatize = lemmatize
        self._tokenize  = tokenize
        self._lowercase = lowercase

        self._nlp = get_spacy_small('en_core_web_sm')
        self._wml = WordNetLemmatizer()

        # warmup
        self._wml.lemmatize('dog')

    def __call__(self, text: str) -> List[str]:
        if not self._active:
            return text.split()

        if self._lowercase and (not self._lemmatize):
            text = text.lower()

            if not self._tokenize:
                text = text.split()

        if self._tokenize and (not self._lemmatize):
            doc  = self._nlp(text)
            text = [t.text for t in doc]

        if self._lemmatize:
            doc  = self._nlp(text)
            text = [t.lemma_ for t in doc]

            # return [self._wml.lemmatize(w) for w in text.split()]

        return text


class Normalizer(Tokenizer):
    """ Usefull to tokenize / lowercase / lemmatize a piece of text
    Returns transformed string, not a list of tokens.
    """

    def __call__(self, text: str) -> str:
        return ' '.join(Tokenizer.__call__(self, text))


# noinspection PyPep8Naming
def Split():
    return Tokenizer()


# noinspection PyPep8Naming
def Lemmatize():
    return Tokenizer(lemmatize=True)


# noinspection PyPep8Naming
def Lower():
    return Tokenizer(lowercase=True)


# noinspection PyPep8Naming
def Tokenize():
    return Tokenizer(tokenize=True)


# noinspection PyPep8Naming
def LowerTokenize():
    return Tokenizer(tokenize=True, lowercase=True)
