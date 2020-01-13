# TODO(elan): we use spacy.load('en') here even though we technically allow other languages in Config.

import logging
import random
import re
from collections import defaultdict
from itertools import chain
from typing import Dict, List
from typing import Type

from pyzsl.data.wiki.config import Config
from pyzsl.data.wiki.src.dict_keys import Entry, DefinitionKeys, EntryKeys
from pyzsl.utils.general import json_iterator, restricted_pickle_load
from pyzsl.utils.nlp import nlp_large, StrList_O, DocItem_T, _is_date, is_stop, \
    is_span, parse, n_words, to_chunks, truncate, Normalizer
from spacy.tokens import Token

DEF_DELIM     = ' EOD '   # end-of-definition
TOK_DEF_DELIM = ' EODT '  # end-of-subtoken-definition

cfg_T = Type[Config.Definitions]


# noinspection PyPep8Naming
def DefinitionGetter(cfg: cfg_T, **kwargs) -> 'DefinitionGetterBase':

    """ Returns an appropriate DefinitionGetter
    by inspecting Config.Definitions.Source
    """

    src = cfg.source
    cls = {
        'wikipedia':  WikiGetter,
        'dictionary': DictionaryGetter,
    }[src]

    return cls(cfg, **kwargs)


class DefinitionGetterBase:
    """ A base class that is responsible for creating definitions for
    categories.

    Subclasses should implement the `_query(s: str)` method, which should return
    a list of available defintions for string `s`, or `None` if no definition
    is available.
    """

    def __init__(self, cfg: cfg_T, debug=False):
        self.cfg         = cfg
        self._processed  = set()
        self._nlp        = nlp_large()
        self._debug_flag = debug     
        self._logger     = logging.getLogger(self.__class__.__name__)

    def _query(self, s: str) -> StrList_O:
        """ Should return a list of available defintions for string `s`,
        or `None` if no definition is available.
        """

        raise NotImplementedError

    def _debug(self, msg):
        """ Print debugging information if flag debug=True was passed to the
        constructor
        """

        if self._debug_flag:
            self._logger.debug(msg)

    def _query_token(self, t: Token) -> StrList_O:
        """ Return definition for a single spacy Token `t`.

        * Skips stopwords
        * Skips dates and numbers
        * Tries to find definition for [text, text.lower(), text.lemma]
        """

        if is_stop(t) or _is_date(t):
            self._debug(f'Token {t} skipped')
            return []

        self._debug(f'Querying token {t}')

        d = self._query(t.text)
        d = d or self._query(t.text.lower())
        d = d or self._query(t.lemma_)

        return d

    def _query_item(self, item: DocItem_T) -> StrList_O:
        """ Return definition for a single spacy Token or spacy Span `item`.
        `item` can be a `Span` if it was marked as a `Named Entity`.

        * Tries to call _query_token
        * Tries to call _query_token for each sub-token in item
        * If the latter, we want all sub-tokens to have definitions
        * If this fails, return None
        * The definitions are merged using `TOK_DEF_DELIM`
        """

        d = self._query_token(item)

        self._debug(f'Querying item {item}')

        if (d is None) and is_span(item):
            try:
                sub_defs = [self._query_token(sub_token)
                            for sub_token in item]
                d        = [TOK_DEF_DELIM.join(item)
                            for item in zip(*sub_defs)]
            except TypeError:
                d = None

        return d

    def _define(self, name: str) -> List[str]:
        """ Generate definitions for category `name`.

        * Tries to query name directly with the source
        * Tries to query [name.lower, name.lemma]
        * Tries to query each item in nlp(name)
        # TODO(elanmart): improve docs.
        """

        doc = self._nlp(name)
        ret = None

        self._debug(f'Trying to define: name = {name}, doc = {doc}')

        ret = ret or self._query(name)
        ret = ret or self._query(name.lower())
        ret = ret or self._query(' '.join(t.lemma_ for t in doc))

        if ret is not None:
            return ret

        # individual tokens lookup
        try:
            tokens = parse(doc)
            defs   = [self._query_item(item)
                      for item in tokens]
            defs   = [d for d in defs 
                      if len(d) > 0]
            ret    = [TOK_DEF_DELIM.join(item)  # TODO(elan) add comment / refactor
                      for item in zip(*defs)]

            if self._debug_flag:
                lengths = [len(d) for d in defs]
                ziplen  = len(list(zip(*defs)))
                self._debug(f'Definition lengths: {lengths}, ziplen: {ziplen}')

            return ret

        except TypeError:
            return []

    def __call__(self, entry: Dict[str, str]) -> Dict[str, List[str]]:
        """ Given a document `entry`, extract definitions for every
        category in entry['categories'] that have not yet been processed.

        Returns None if no new category was found in the document.
        Else returns a mapping {category_name => list_of_category_definitions}
        """

        entry  = Entry(entry)
        labels = [name for name in entry.categories
                  if name not in self._processed]

        # generate definitions.
        ret = {name: self._define(name)
               for name in labels}

        self._processed |= set(labels)

        return ret or None


class DictGetterBase(DefinitionGetterBase):
    """ Creates definitions by using a dictionary that can be found at data_path
    Derived classes should define a method `_load` loading the dictionary
    from data_path into `_db` attribute.
    """

    def __init__(self, cfg, data_path):
        super().__init__(cfg)
        self._db = self._load(data_path)

    def _load(self, path: str) -> Dict[str, List[str]]:
        """ Load the dictionary from data_path into `self._db` attribute.
        """

        raise NotImplementedError

    def _handle_multiple_definitions(self, available: List[str]) ->List[str]:
        """ Handle the case when multiple definitions are available.
        See Config.DictConfig.mode for details.

        Probably nothing else than `all` or `first` makes sense.
        """

        if self.cfg.DictConfig.mode == 'all':
            return available

        if self.cfg.DictConfig.mode == 'first':
            return [
                available[0]
            ]

        if self.cfg.DictConfig.mode == 'concat':
            return [
                DEF_DELIM.join(available)
            ]

        if self.cfg.DictConfig.mode == 'longest':
            return [
                max(available, key=len)
            ]

        if self.cfg.DictConfig.mode == 'random':
            return [
                random.choice(available)
            ]

        if self.cfg.DictConfig.mode == 'first-and-random':
            ret = available[0]

            if len(available) > 1:
                ret = ret + DEF_DELIM + random.choice(available[1:])

            return [
                ret
            ]

        raise ValueError(f"DictConfig.mode == {self.cfg.DictConfig.mode} not recognized!")

    def _query(self, s: str) -> StrList_O:
        """ See DefinitionGetterBase for details.
        """

        self._debug(f'Query received: {s}')

        available = self._db.get(s)  # type: StrList_O

        if available is None:
            return None

        elif isinstance(available, str):
            return [available]

        else:
            return self._handle_multiple_definitions(available)


class WikiGetter(DictGetterBase):
    """ Generates definitions using wikipedia dump itself.
    Specifically, looks for documents where title matches the requested
    name.
    """

    def _load(self, path: str) -> Dict[str, str]:
        """ Load the dictionary from data_path into `self._db` attribute.
        """

        self._logger.info(f'Loading wikipedia definitions from {path}.')

        disambig   = re.compile(r'\(.+?\)')
        dictionary = {}

        for row in json_iterator(path):
            title  = row['title']
            fields = [row[k] for k in self.cfg.WikiConfig.fields]

            if self.cfg.WikiConfig.title_lower:
                title = title.lower()

            if self.cfg.WikiConfig.title_lemma:
                title = ' '.join([token.lemma_ for token in self._nlp(title)])

            if self.cfg.WikiConfig.remove_parenthesis:
                title = disambig.sub('', title).strip()

            dictionary[title] = ' <EOD> '.join(fields)

        self._logger.info(f'Wikipedia read, total of {len(dictionary):_} '
                          f'keys were retrieved.')

        return dictionary


class DictionaryGetter(DictGetterBase):
    """ Generates definitions using a dictionary given as a pickled python object
    in a format [(key, definition), (key, definition)] where keys dont have to
    be unique.

    Specifically, we use dictionary available at
    http://www.cl.cam.ac.uk/~fh295/dicteval.html
    """

    def _define(self, s: str):
        """ Overrides _define method, because the dictionary sometimes stores
        names such as 'non-verbal' as 'nonverbal', so we want to remove
        the dash.
        """

        s = s.lower().replace('-', '')
        return DictGetterBase._define(self, s)

    def _load(self, path: str) -> Dict[str, List[str]]:
        """ Load the dictionary from data_path into `self._db` attribute.
        """

        self._logger.info(f'Getting pickled dictionary from {path}')

        ret = defaultdict(list)
        dictionary = restricted_pickle_load(path)

        for k, v in zip(*dictionary):
            ret[k].append(' '.join(v))

        self._logger.info(f'Dictionary read, total of {len(ret):_} '
                          f'keys were retrieved.')

        return dict(ret)


class EmptyValidator:
    """ Return only the categories for which a non-empty definition list is
    available.

    If no such items are found, return None
    """

    def __call__(self, item):
        ret = {k: v
               for k, v in item.items()
               if len(v) > 0}

        if len(ret) == 0:
            return None

        return ret


class DefinitionLengthValidator:
    """ Retain only the definitions that meet the length criteria.

    Shorter than allowed are discarded, longer are trimmed or split.
    You probably want to use trimming tho.
    """

    def __init__(self, min_length, max_length, split):
        self.min_length = min_length
        self.max_length = max_length
        self.split      = split

    def __call__(self, entries):
        ret = {}
        for label_name, label_definitions in entries.items():

            if self.split:
                label_definitions = [
                    chunk
                    for d in label_definitions
                    for chunk in to_chunks(d, self.max_length)
                    if n_words(d) > self.min_length
                ]

            else:
                label_definitions = [
                    truncate(d, self.max_length)
                    for d in label_definitions
                    if n_words(d) > self.min_length
                ]

            ret[label_name] = label_definitions

        return ret


class LabelUsedValidator:
    """ This will retain only those labels (categories) that are
    assigned to at least one item in any of the `doc_files`.

    Doc_files should be the documents in the same format as the documents
    extracted from wikipedia dump.
    """

    def __init__(self, *doc_files):
        K = EntryKeys
        
        iterators  = [json_iterator(fname) for fname in doc_files]
        self._used = set()
        
        for row in chain(*iterators):
            self._used |= set(row[K.categories])

    def __call__(self, item):
        K = DefinitionKeys

        if item[K.name] in self._used:
            return item

        return None


class DefinitionNormalizer:
    """ Apply pyzsl.utils.nlp.Normalizer to item['definition'].

    TODO(elanmart): do we need a custom tokenizer here?
    """
    def __init__(self, tokenize, lowercase, lemmatize, keys=()):
        self._tokenizer = Normalizer(tokenize=tokenize, lowercase=lowercase, lemmatize=lemmatize)

    def __call__(self, item):
        K = DefinitionKeys
        d = item[K.definition]
        item[K.definition] = self._tokenizer(d)

        return item


class SchemaChanger:
    """ Given definitions for multiple categories, repackage them to a different
    python collection.

    Specifically, given list [{name_1: defs_1}, {name_2: defs_2}]
    return [{'name': name_1, 'definition': defs_1[0]},
            {'name': name_1, 'definition': defs_1[1],
            ...
            {'name': name_2, 'definition': defs_2[0],
            ...]

    """
    def __call__(self, item):
        K = DefinitionKeys

        item[K.name]: str
        item[K.definition]: List[str]

        ret = [
            {
                K.name: k,
                K.definition: d
            }
            for k, def_list in item.items()
            for d in def_list
        ]

        return ret or None
