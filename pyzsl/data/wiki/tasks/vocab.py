""" Tasks associated with building vocabularies on the extracted data.

Currently we provide tasks to build
    * Vocabulary from documents and definitions seperately (two different vocabs)
    * Vocabylary from documents and definitions jointly.
"""

import luigi
from luigi.util import requires, inherits

from pyzsl.data.wiki.src.dict_keys import EntryKeys, DefinitionKeys
from pyzsl.data.wiki.src.vocabs import Vocab
from pyzsl.data.wiki.tasks.core import WikiTask
from pyzsl.data.wiki.tasks.middle import SeenUnseenFinal
from pyzsl.utils.general import json_load, dill_dump, load_zsl_split
from pyzsl.utils.luigi import Wrapper
from pyzsl.utils.nlp import Split, Lemmatize


@requires(SeenUnseenFinal)
class VocabBuilderBase(WikiTask):
    def get_vocab(self):
        raise NotImplementedError

    def get_path(self):
        raise NotImplementedError

    def run(self):
        vocab = self.get_vocab()
        path  = self.get_path()

        vocab.build()
        dill_dump(vocab, path)

    def output(self):
        return luigi.LocalTarget(self.get_path())


class JointVocabBuilder(VocabBuilderBase):

    def get_path(self):
        return self.paths.joint_vocab

    def get_vocab(self):

        return Vocab(
            max_size=None,
            sources={
                self.paths.json_data.train: {
                    EntryKeys.title:   Split(),
                    EntryKeys.summary: Lemmatize()
                },

                self.paths.json_data.definitions: {
                    DefinitionKeys.name: Lemmatize(),
                    DefinitionKeys.definition: Split()
                }
            }
        )


class DocVocab(VocabBuilderBase):

    def get_path(self):
        return self.paths.disjoint_vocab_doc

    def get_vocab(self):
        K = EntryKeys

        return Vocab(
            max_size=None,
            sources={
                self.paths.json_data.train: {
                    K.title:   Lemmatize(),
                    K.summary: Lemmatize()
                },
            }
        )


class DefVocab(VocabBuilderBase):

    def get_path(self):
        return self.paths.disjoint_vocab_def

    def get_vocab(self):
        K       = DefinitionKeys
        seen, _ = load_zsl_split(self.paths.zsl_split)
        cond    = lambda item: item[K.name] in seen

        return Vocab(
            max_size   = None,
            condition= cond,
            sources    = {
                self.paths.json_data.definitions: {
                    K.name: Lemmatize(),
                    K.definition: Split()
                }
            },
        )


@inherits(VocabBuilderBase)
class Vocabs(Wrapper, WikiTask):
    tasks = [
        JointVocabBuilder,
        DocVocab,
        DefVocab,
    ]
