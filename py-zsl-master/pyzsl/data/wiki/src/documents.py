# TODO(elan): change all signatures to take explicit arguments instead of Config classes
# TODO(elan): figure out if we indeed want to generate any additional fields

import numpy as np

from pyzsl.data.wiki.src.dict_keys import DefinitionKeys, EntryKeys
from pyzsl.utils.nlp import n_words, truncate, Normalizer
from pyzsl.utils.general import json_iterator


class LengthValidator:
    """ Returns None if an entry does not meet the length requirements.

    Parameters
    ----------
    min_length: int
        Minimum length (in words) of the considered text
        to accept an entry as valid.
    fields: List[str]
        List of fields we should take into consideration when measuring
        length of the text.
    mode: str
        'all' or 'concat'.
        If 'all', every field in `fields` must have length >= min_length
        If `concat`, concatenated fields must have length >= min_length.
    """

    def __init__(self, min_length, fields, mode):
        self._min_length = min_length
        self._fields     = fields
        self._mode       = mode

        if mode not in {'all', 'concat'}:
            raise RuntimeError(f'Mode <{mode}> not recognized '
                               f'in class {self.__class__.__name__}')

    def __call__(self, entry):
        """
        Parameters
        ----------
        entry : Dict
            Entry representing one wikipedia article.
        """

        items = [entry[f]
                 for f in self._fields]

        if self._mode == 'all':
            condition = all(n_words(text) >= self._min_length
                            for text in items)

        elif self._mode == 'concat':
            condition = sum(n_words(text)
                            for text in items) >= self._min_length

        else:
            raise RuntimeError()

        if condition is True:
            return entry

        return None


class LabelValidator:
    """ Used to remove from the list of categories those entries, that do
    not have a definition in `definitions` file.

    Return `None` if after this 'prining',
    entry does not have any categories assigned.

    Parameters
    ----------
    definitions : str
        path to a file where definitions are stored.
    """

    def __init__(self, definitions):
        self._labels = set()

        K = DefinitionKeys
        for row in json_iterator(definitions):
            name = row[K.name]
            self._labels.add(name)

    def __call__(self, entry):
        """
        Parameters
        ----------
        entry : Dict
            Entry representing one wikipedia article.
        """

        K = EntryKeys
        entry[K.categories] = [name
                               for name in entry[K.categories]
                               if name in self._labels]

        if len(entry[K.categories]) == 0:
            return None

        return entry


class FieldExtractor:
    """ Extracts only certain fields from the wikipedia document, discarding the rest.

    Parameters
    ----------
    text_fields : List[str]
        Fields to extract and potentially concatenate / transform
    meta_fields : List[str]
        Fields to extract and pass to the output without changing.
    merge_textfields : bool
        If true, merge the extracted text_fields into one.
        Its name will be a concatenation of `text_fields` using
        underscore: `field1_field2_field3...`
    merge_token : str
        token used to seperate text from different text fields.
        If `None`, 'EO_<fieldname>' will be used.
    """

    def __init__(self, text_fields, meta_fields=(), merge_textfields=False, merge_token=None):
        self._text_flds = text_fields
        self._meta_flds = meta_fields
        self._merge     = merge_textfields

        if merge_token is None:
            self._merge_tokens = [f' EO_{f}' for f in self._text_flds]
        else:
            self._merge_tokens = [merge_token] * len(self._text_flds)

    def __call__(self, entry):
        """
        Parameters
        ----------
        entry : Dict
            Entry representing one wikipedia article.
        """

        entry = {k: entry[k]
                 for k in self._text_flds + self._meta_flds}

        if self._merge:
            key    = '_'.join(self._text_flds)
            value  = ' '.join(entry[f] + tok
                              for f, tok in zip(self._text_flds,
                                                self._merge_tokens))

            entry    = {
                key: value,
                **{k: entry[k] for k in self._meta_flds}
            }

        return entry


class Truncator:
    """ Trims fields to have at most `max_length` words 
    """
    
    def __init__(self, max_legnth, fields):
        self._max_length = max_legnth
        self._fields     = fields

    def __call__(self, entry):
        """
        Parameters
        ----------
        entry : Dict
            Entry representing one wikipedia article.
        """
        
        for f in self._fields:

            v = entry[f]
            v = truncate(v, self._max_length)
            entry[f] = v

        return entry


class Splitter:
    """ Splits the incoming entries into train, dev, and test sets
    according to the tuple of provided probabilities `p`

    Parameters
    ----------
    p: Tuple[float, float, float]
        Probabilities of sending an item to, respectively,
        a train, development (validation), or test set.
    """

    def __init__(self, p):
        self._i  = np.array([0, 1, 2])
        self._p  = np.array(p)

        self._prob = self._reset()
        self._iter = self._iter_fn()

    def _reset(self):
        self._prob = np.random.choice(self._i, size=10_000, replace=True, p=self._p)
        return self._prob

    def _iter_fn(self):
        idx = 0
        while True:

            if idx == self._prob.size:
                self._reset()
                idx = 0

            ret  = self._prob[idx]
            idx += 1

            yield ret

    def __call__(self, entry):
        retval      = [None, None, None]
        idx         = next(self._iter)
        retval[idx] = entry

        return tuple(retval)


class DocumentTokenizer:
    """ Runs pyzsl.utils.nlp.Normalizer(tokenize, lowercase, lemmatize) on
    each field in `keys`.
    """

    def __init__(self, keys, tokenize=False, lowercase=False, lemmatize=False):
        self._keys      = keys
        self._tokenizer = Normalizer(tokenize=tokenize, lowercase=lowercase, lemmatize=lemmatize)

    def __call__(self, item):
        for k in self._keys:
            d = item[k]
            d = self._tokenizer(d)
            item[k] = d

        return item
