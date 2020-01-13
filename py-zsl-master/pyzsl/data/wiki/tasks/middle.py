""" Picks up where core.py left off. Contains tasks that can take
the output of WikiExtractor and generate final, human-readable (json)
splits of the data:
    * train / dev / test documents
    * label definitions
    * seen / unseen label splits
"""

import logging

import luigi
from luigi.util import requires, inherits

from pyzsl.data.wiki.config import Config
from pyzsl.data.wiki.src.definitions import DefinitionGetter, SchemaChanger, \
    EmptyValidator, DefinitionLengthValidator, LabelUsedValidator, \
    DefinitionNormalizer
from pyzsl.data.wiki.src.documents import LengthValidator, LabelValidator, \
    FieldExtractor, Truncator, Splitter, DocumentTokenizer as DocTokenizer
from pyzsl.data.wiki.src.labels import make_label_split, LabelSplitter, \
    make_label_vocabs
from pyzsl.data.wiki.src.mappers import JsonMapper
from pyzsl.data.wiki.tasks.core import ExtarctTask, WikiTask
from pyzsl.utils.general import mktemp, json_dump, numpy_dump
from pyzsl.utils.luigi import local_targets, Wrapper
from pyzsl.utils.nlp import shuf, append_


@requires(ExtarctTask)
class TmpDefinitionsTask(WikiTask):
    """ Creates definitions for categories found in the dataset, and writes them to temporary file.

    File is temporary as we will apply further transforms and filters before the definitions are
    ready to be used.

    This will read the extracted wikipedia, and run the following transforms:
    * DefinitionGetter          ->
    * DefinitionLengthValidator ->
    * EmptyValidator            -> remove definitions that
    * SchemaChanger             -> transform definitions

    Notes
    -----
    Task Parameters:
        The generated definitions are parametrized by `Config.Definitions`

    Adding new definitions source:
        * inherit from `DefinitionGetterBase` in `.src.definitions`.
        * add an entry to `.src.definitions.DefinitionGetter()`
        * add an entry to `_get_data_path`

    Modyfing pre / post - processing:
        * Add / Remove entries to the `JsonMapper.apply` call in the `run` method.

    """

    def _get_data_path(self, source):

        tmp_documents = [
            self.paths.tmp_data.train,
            self.paths.tmp_data.dev,
            self.paths.tmp_data.test,
        ]

        return {
            'dictionary': self.paths.dictionary,
            'wikipedia': tmp_documents
        }[source]

    def run(self):
        logger = logging.getLogger(self.__class__.__name__)

        src = self.paths.dump_json
        data = self._get_data_path(Config.Definitions.source)

        min_ = Config.Definitions.Length.low
        max_ = Config.Definitions.Length.high
        spl_ = Config.Definitions.Policy.split

        logger.info(
            f'Running definition generation from {src} to {self.output().path} '
            f'given source <{Config.Definitions.source}> '
            f'available at {data}, for params(min={min_}, max={max_}, split={spl_}. ')

        with self.tmp_output() as dest:
            JsonMapper(src, dest).apply(
                DefinitionGetter(Config.Definitions, data_path=data),
                DefinitionLengthValidator(min_length=min_, max_length=max_, split=spl_),
                EmptyValidator(),
                SchemaChanger(),
            )

    def output(self):
        output = self.paths.tmp_data.definitions
        return luigi.LocalTarget(output)


@requires(TmpDefinitionsTask)
class TmpDocumentsTask(WikiTask):
    """ Creates temporary {train,dev,test} files with documents extracted from whole dataset.

    File is temporary as we will apply further transforms and filters before the documents are
    ready to be used.

    This will read the extracted wikipedia, and run the following transforms:
        * LengthValidator -> discard too short documents
        * LabelValidator  -> discard documents without labels
        * Truncator       -> trim the documents to the desired length
        * FieldExtractor  -> discard unused fields
        * Splitter        -> randomly stream the entry to one of {train,dev,test} files

    Notes
    -----
    Task Parameters:
        The generated definitions are parametrized by `Config.Documents`

    Modyfing pre / post - processing:
        * Add / Remove entries to the `JsonMapper.apply` call in the `run` method.
    """

    def run(self):
        with self.tmp_output() as (train, dev, test):

            src = self.paths.dump_json
            cfg = Config
            cfg_doc = cfg.Documents
            cfg_len = cfg_doc.Length

            JsonMapper(src, train, dev, test).apply(
                LengthValidator(min_length=cfg_len.low, fields=cfg_len.used_fields, mode=cfg_len.conditions),
                LabelValidator(definitions=self.paths.tmp_data.definitions),
                Truncator(max_legnth=cfg_len.high, fields=cfg_doc.fields),
                FieldExtractor(text_fields=cfg_doc.fields,
                               meta_fields=['categories'],
                               merge_textfields=cfg_doc.merge_fields,
                               merge_token=cfg_doc.merge_token),
                Splitter(p=cfg.Split.proportions)
            )

    def output(self):
        _, train, dev, test = self.paths.tmp_data

        return local_targets([
            train,
            dev,
            test
        ])


@requires(TmpDocumentsTask)
class SplitTask(WikiTask):
    """ Generates a split of 'seen' vs 'unseen' labels.
    """

    def run(self):
        with self.tmp_output() as seen_unseen:
            tmp_defs, *tmp_documents = self.paths.tmp_data

            make_label_split(doc_path=tmp_documents,
                             def_path=tmp_defs,
                             min_freq=Config.UnseenLabels.FreqnecyConfig.min_freq,
                             output_path=seen_unseen)

    def output(self):
        return luigi.LocalTarget(self.paths.tmp_zsl_split)


@requires(SplitTask)
class DocJsonGeneratorTask(WikiTask):
    """ Generate final {train,dev,test}.json files.

    This means moving all the examples with unseen labels from train to test file,
    and running optional postprocessing.

    You can modify the postprocessing by altering entries in
    Config.Documents.PostProc
    """

    def run(self):
        with self.tmp_output() as (final_train, final_dev, final_test):

            (_,
             tmp_train,
             tmp_dev,
             tmp_test) = self.paths.tmp_data

            cfg_pproc = Config.Documents.PostProcessing
            splitter  = LabelSplitter(self.paths.tmp_zsl_split)

            postproc = [
                DocTokenizer(lowercase=cfg_pproc.lower, tokenize=cfg_pproc.token,
                             lemmatize=cfg_pproc.lemma, keys=cfg_pproc.fields)
            ]

            with mktemp() as train_buffer, mktemp() as test_buffer:
                # train examples and moved-to-test examples
                JsonMapper(tmp_train, train_buffer, test_buffer).apply(
                    *postproc,
                    splitter
                )

                JsonMapper(tmp_dev, final_dev).apply(
                    *postproc
                )

                JsonMapper(tmp_test, final_test).apply(
                    *postproc
                )

                shuf(f_in=train_buffer, f_out=final_train)
                append_(dest=final_test, sources=[test_buffer])

    def output(self):
        return local_targets([
            self.paths.json_data.train,
            self.paths.json_data.dev,
            self.paths.json_data.test,
        ])


@requires(DocJsonGeneratorTask)
class DefJsonGeneratorTask(WikiTask):
    """ Generate final definitions.json file. This means
    removing labels that did not appear in the final data,
    as well as running some processing on the text of the definitions.

    You can modify the postprocessing by altering entries in
    Config.Definitions.PostProc  # TODO(elanmart): actually add this.
    """

    def run(self):
        with self.tmp_output() as output:

            cfg_pproc = Config.Definitions.PostProcessing
            defs, *docs = self.paths.tmp_data

            postproc = [
                DefinitionNormalizer(tokenize=cfg_pproc.token,
                                     lemmatize=cfg_pproc.lemma,
                                     lowercase=cfg_pproc.lower),
            ]

            JsonMapper(defs, output).apply(
                LabelUsedValidator(*docs),
                *postproc
            )

    def output(self):
        return luigi.LocalTarget(self.paths.json_data.definitions)


@inherits(SplitTask)
class JsonTask(Wrapper, WikiTask):
    """ Wrapper task used to generate both
        * Document json files
        * Definition json files
    """

    tasks = [
            DocJsonGeneratorTask,
            DefJsonGeneratorTask,
        ]


@requires(JsonTask)
class YsVocab(WikiTask):
    def run(self):
        (stoi,
         itos,
         labels,
         intervals,
         zsl_split) = make_label_vocabs(self.paths.json_data)

        with self.tmp_output() as (stoi_path, itos_path, def_labels_path, intervals_path, zsl_split_path):
            json_dump(stoi,       stoi_path)
            json_dump(itos,       itos_path)
            json_dump(zsl_split,  zsl_split_path)
            numpy_dump(labels,    def_labels_path)
            numpy_dump(intervals, intervals_path)

    def output(self):
        return local_targets([
            self.paths.label_stoi,
            self.paths.label_itos,
            self.paths.definition_labels,
            self.paths.intervals,
            self.paths.zsl_split,
        ])


@inherits(YsVocab)
class SeenUnseenFinal(Wrapper, WikiTask):
    tasks = [
        YsVocab
    ]  # TODO(elanmart): this has to go. Refactor prev. task to a number of sub-tasks.
