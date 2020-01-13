import array
from collections import defaultdict
from collections import namedtuple
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from pyzsl.metrics import scores_to_topk
from pyzsl.models.wrappers.base import BaseWrapper
from pyzsl.utils.general import CsrBuilder, json_iterator, sparse_div_
from pyzsl.utils.nlp import get_spacy


class LabelInDoc(BaseWrapper):

    ReturnType = namedtuple('ReturnType', ['unigram_matches', 'direct_matches', 'doc_lengths'])

    class Modes:
        name     = 'name'
        document = 'document'

    def __init__(self,
                 mode='name',
                 normalize=True,
                 disable_tagger=True,
                 lower_doc=True, lemma_doc=True,
                 lower_name=True, lemma_name=True,
                 remove_punct=True, remove_stop=True,
                 batch_size=4096):

        super().__init__()

        assert mode in {self.Modes.name, self.Modes.document}

        self.mode      = mode
        self.normalize = normalize

        self.disable_tagger = disable_tagger

        self.lower_doc = lower_doc
        self.lemma_doc = lemma_doc

        self.lower_name = lower_name
        self.lemma_name = lemma_name

        self.remove_punct = remove_punct
        self.remove_stop  = remove_stop

        self._nlp = get_spacy('en_core_web_sm')
        self._nlp.disable_pipes('parser', 'ner')

        if self.disable_tagger:
            self._nlp.disable_pipes('tagger')

        self.lengths_ = []
        self.mapping_ = defaultdict(list)

        self._batch_size = batch_size

    def _name_to_tokens(self, name):

        retval = []

        if self.lower_name:
            name = name.lower()

        for token in self._nlp(name):

            if self.remove_stop and token.is_stop:
                continue

            if self.remove_punct and token.is_punct:
                continue

            if self.lemma_name:
                token = token.lemma_
            else:
                token = token.text

            retval.append(token)

        return retval

    def _doc_to_tokens(self, doc):

        if self.lower_doc:
            doc = doc.lower()

        if self.lemma_doc:
            return [token.lemma_ for token in self._nlp(doc)]
        else:
            return [token.text for token in self._nlp(doc)]

    def fit(self, definitions: str):
        self.lengths_ = []
        self.mapping_ = defaultdict(list)

        for idx, row in enumerate(json_iterator(definitions)):

            name = row['name']
            self.lengths_.append(0)

            for token in self._name_to_tokens(name):
                self.mapping_[token].append(idx)
                self.lengths_[idx] += 1

        self.lengths_ = np.array(self.lengths_)
        self.mapping_ = {
            k: array.array('i', v)
            for k, v in self.mapping_.items()
        }

    def get_exact_matches(self, path):
        raise NotImplementedError("This should implement the exact match btween label name and document "
                                  "But this is an exhaustive procedure.")

    def get_unigram_matches(self, path: str):

        builder = CsrBuilder(dtype=np.float32,  remove_duplicates=True)

        def supply_batch():
            nonlocal builder

            ret     = builder.get(n_cols=len(self.lengths_))
            builder = CsrBuilder(dtype=np.float32,  remove_duplicates=True)

            if (self.mode == self.Modes.name) and self.normalize:
                # noinspection PyTypeChecker
                ret = sparse_div_(ret, self.lengths_)

            return ret

        for row_idx, row in enumerate(json_iterator(path), 1):

            # extract
            words = row['summary']  # TODO(elanmart) why only summary? Why use string as key in __getitem__?
            words = self._doc_to_tokens(words)

            # foobar
            if self.mode == self.Modes.name:

                words = {w for w in words if (w in self.mapping_)}
                for w in words:
                    inds = self.mapping_[w]
                    data = [1] * len(inds)
                    builder.update_row(indices=inds, data=data)

            else:
                raise NotImplementedError()

            builder.row_finished()

            if (row_idx % self._batch_size) == 0:
                yield supply_batch()

        else:
            yield supply_batch()

    def predict_topk(self, path, k=1, randomize=True):
        with open(path) as f:
            N = sum(1 for _ in f)

        retval = np.zeros((N, k), dtype=np.int32)
        idx    = 0

        def noise(shape, magnitude):
            return (np.random.rand(*shape) - 0.5) * 2 * magnitude

        for csr in self.get_unigram_matches(path):

            if randomize:
                csr.data += noise(csr.data.shape, 1e-3)

            topk = scores_to_topk(csr.toarray(), k=k)
            nxt  = idx + csr.shape[0]

            retval[idx:nxt, :] = topk
            idx += csr.shape[0]

        return retval

    def predict_scores(self, path: str) -> Tuple[sp.csr_matrix, sp.csr_matrix]:

        (unigram_matches,
         doc_lengths)     = self.get_unigram_matches(path)
        exact_matches     = self.get_exact_matches(path)

        if self.normalize_by_doc:
            unigram_matches  = sparse_div_(unigram_matches, doc_lengths.reshape(-1, 1))

        if self.normalize_by_name:
            unigram_matches = sparse_div_(unigram_matches, self.lengths_)

        return unigram_matches, exact_matches

