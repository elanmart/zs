""" Contains tasks to transform the dataset from json format into various
more useful representations. For now this includes
    * Tensors
    *
    *
"""

import luigi
import numpy as np
from luigi.util import inherits, requires

from pyzsl.data.wiki.src.dict_keys import EntryKeys, DefinitionKeys
from pyzsl.data.wiki.src.formats import tensor_to_csr, csr_to_tfidf, \
    index_to_fasttext, write_starpace_basedoc, write_starpace_file, name_vectors
from pyzsl.data.wiki.src.labels import get_label_matrix, seen_definitions_mask
from pyzsl.data.wiki.src.vocabs import Vocab
from pyzsl.data.wiki.tasks.core import WikiTask
from pyzsl.data.wiki.tasks.middle import SeenUnseenFinal, YsVocab
from pyzsl.data.wiki.tasks.vocab import JointVocabBuilder, DocVocab, DefVocab, \
    Vocabs
from pyzsl.utils.general import dill_load, numpy_dump, scipy_dump, numpy_load
from pyzsl.utils.luigi import local_targets, temp_path, Wrapper
from pyzsl.utils.nlp import vec_large


@requires(Vocabs)
class YTensorTask(WikiTask):
    """ This will generate scipy.csr_matrix of lables for each split in [train, dev, test].
    Each row `i` contains labels / categories associated with `i`-th example.
    The rows are sparse, and `row[j]` is `1` if `j`-th label is assigned to this example, else `0`.
    """

    def run(self):
        defs, *docs = self.paths.json_data
        for path_in, output in zip(docs, self.output()):

            y = get_label_matrix(path_in, self.paths.label_stoi)
            _ = scipy_dump(y, output.path)

    def output(self):
        return local_targets([
            self.paths.Y_csr_matrices.train,
            self.paths.Y_csr_matrices.dev,
            self.paths.Y_csr_matrices.test,
        ])


@inherits(YTensorTask)
class DocTensor(WikiTask):
    """ Builds a tensor representation of the documents, given the Vocabulary
    specified in VocabCls.

    Each row contains indices of words in the document, padded with 0s if
    document is shorter than max. length in the dataset.
    """

    def get_vocab_task(self):
        raise NotImplementedError

    def get_paths(self):
        raise NotImplementedError

    def run(self):
        with self.tmp_output() as output_paths:
            defs, *docs = self.paths.json_data
            K           = EntryKeys
            vocab       = dill_load(self.input().path)
            field_info  = [
                K.title,
                K.summary
            ]

            for path_in, path_out in zip(docs, output_paths):
                indices = vocab.transform_file(path_in, field_info)
                size    = max(len(row) for row in indices)
                array   = np.zeros((len(indices), size), dtype=np.int64)

                for i, row in enumerate(indices):
                    array[i, :len(row)] = row

                numpy_dump(array, path_out)

    def requires(self):
        VocabTask = self.get_vocab_task()
        return VocabTask(**self.param_kwargs)

    def output(self):
        return local_targets(
            self.get_paths()
        )


@inherits(SeenUnseenFinal)
class DefTensor(WikiTask):
    """ Builds a tensor representation of the definitions, given the Vocabulary
    specified in VocabCls.

    Each row contains indices of words in the document, padded with 0s if
    document is shorter than max. length in the dataset.
    """

    def get_vocab_task(self):
        raise NotImplementedError

    def get_path(self):
        raise NotImplementedError

    def run(self):
        K          = DefinitionKeys
        vocab      = dill_load(self.input().path)  # type: Vocab
        field_info = [
            K.name,
            K.definition
        ]

        indices = vocab.transform_file(self.paths.json_data.definitions, field_info=field_info)
        size    = max(len(item) for item in indices)
        array   = np.zeros((len(indices), size), dtype=np.int64)

        for i, row in enumerate(indices):
            array[i, :len(row)] = row

        with self.tmp_output() as path_out:
            numpy_dump(array, path_out)

    def requires(self):
        VocabTask = self.get_vocab_task()
        return VocabTask(**self.param_kwargs)

    def output(self):
        return luigi.LocalTarget(
            self.get_path()
        )


@inherits(SeenUnseenFinal)
class JointDefTensor(DefTensor):
    """ Builds a tensor representations of definitions
    using a joint vocabulary for documents and definitions.

    See DefTensor for details.
    """

    def get_vocab_task(self):
        return JointVocabBuilder

    def get_path(self):
        return self.paths.index_data_joint.definitions


@inherits(SeenUnseenFinal)
class DisjointDefTensor(DefTensor):
    """ Builds a tensor representations of documents
    using a vocabulary built on definitions only.

    See DefTensor for details.
    """

    def get_vocab_task(self):
        return DefVocab

    def get_path(self):
        return self.paths.index_data_disjoint.definitions


@inherits(SeenUnseenFinal)
class JointDocTensor(DocTensor):
    """ Builds a tensor representations of documents
    using a joint vocabulary for documents and definitions.

    See DocTensor  for details.
    """

    def get_vocab_task(self):
        return JointVocabBuilder

    def get_paths(self):
        return [
            self.paths.index_data_joint.train,
            self.paths.index_data_joint.dev,
            self.paths.index_data_joint.test,
        ]


@inherits(SeenUnseenFinal)
class DisjointDocTensor(DocTensor):
    """ Builds a tensor representations of documents
    using a vocabulary built on documents only.

    See DocTensor for details.
    """

    def get_vocab_task(self):
        return DocVocab

    def get_paths(self):
        return [
            self.paths.index_data_disjoint.train,
            self.paths.index_data_disjoint.dev,
            self.paths.index_data_disjoint.test,
        ]


@inherits(SeenUnseenFinal)
class TensorTasks(Wrapper, WikiTask):
    """ Wrapper task to build all of the above at once.

    This means that requesting this task means requesting
    all the tasks listed in `requires(self)` method.

    This task will trigger the following tasks:

        * YTensorTask        -> sparse binary indicator matrices Y for each set
        * JointDocTensor     -> document representation build using a joint vocabulary (docs + definitions)
        * DisjointDocTensor  -> document representation build using a disjoint vocabulary (only docs)
        * JointDefTensor     -> definition representation build using a joint vocabulary (docs + definitions)
        * DisjointDefTensor  -> definition representation build using a disjoint vocabulary (only definitions)
    """

    tasks = [
            YTensorTask,
            JointDocTensor,
            DisjointDocTensor,
            JointDefTensor,
            DisjointDefTensor,
        ]


@inherits(TensorTasks)
class CsrBaseTask(WikiTask):
    """ Transofrm the representation of dense indices (e.g. [3, 9, 1, 7, 0, 0]) into sparse
    csr matrix.

    This is only changes the representation of the data, no information is changed or lost.
    This means that the CSR matrix still holds information about the word order.

    Inputs
    ------
        * definitions, {train, dev, test}-documents in `Tensor` format (rows of word indices,
        padded with 0s)

    Outputs
    -------
    This will transform index representations of
        * train, dev, test
        * definitions

    You can consider calling
    >>> X = ...
    >>> X.sum_duplicates()
    >>> X.sort_indices()
    """

    DocsTaskCls        = NotImplemented
    DefinitionsTaskCls = NotImplemented
    share_dim          = NotImplemented

    def output_paths(self):
        raise NotImplementedError

    def run(self):
        (def_input,
         doc_input) = self.input()

        (def_output,
         *doc_outputs) = self.output()

        n_cols_def = numpy_load(def_input.path).max() + 1
        n_cols_doc = max(numpy_load(input_.path).max() + 1 for input_ in doc_input)

        if self.share_dim:
            n_cols_def = n_cols_doc = max([n_cols_def, n_cols_doc])

        scipy_dump(
            tensor_to_csr(def_input.path, n_cols=n_cols_def),
            def_output.path
        )

        for input_, output_ in zip(doc_input, doc_outputs):
            scipy_dump(
                tensor_to_csr(input_.path, n_cols=n_cols_doc),
                output_.path
            )

    def output(self):
        return local_targets(
            self.output_paths()
        )

    def requires(self):
        kw = self.param_kwargs
        return [
            self.DefinitionsTaskCls(**kw),
            self.DocsTaskCls(**kw),
        ]


class JointCsrTask(CsrBaseTask):
    DocsTaskCls        = JointDocTensor
    DefinitionsTaskCls = JointDefTensor
    share_dim          = True

    def output_paths(self):
        return [
            self.paths.csr_joint.definitions,
            self.paths.csr_joint.train,
            self.paths.csr_joint.dev,
            self.paths.csr_joint.test,
        ]


class DisjointCsrTask(CsrBaseTask):
    DocsTaskCls        = DisjointDocTensor
    DefinitionsTaskCls = DisjointDefTensor
    share_dim          = False

    def output_paths(self):
        return [
            self.paths.csr_disjoint.definitions,
            self.paths.csr_disjoint.train,
            self.paths.csr_disjoint.dev,
            self.paths.csr_disjoint.test,
        ]


@inherits(CsrBaseTask)
class CsrTasks(Wrapper, WikiTask):
    tasks = [
        JointCsrTask,
        DisjointCsrTask
    ]


@requires(CsrTasks)
class TfIdfTaskBase(WikiTask):
    """ Transofrm the CSR matrices with scikit-learn's TfidfTransformer.

    This is a Base task you should inherit and fill in the abstract methods
    depending on the usecase.

    We provde three basic scenarios
        * When using shared Vocabulary, build the term frequencies on train documents only
            Use it to transform documents and definitions
        * When using shared Vocabulary, build the term frequencies on both train documents and all the definitions
            Use it to transform everyhing (all documents and all definitions)
        * Using separate vocabularies, built separate term freqs from train docs and train defitions
            Use them to transform all documents, and all definitions, respectively.

    """

    def to_fit(self):
        raise NotImplementedError

    def to_transform(self):
        raise NotImplementedError

    def output_paths(self):
        raise NotImplementedError

    def input_masks(self):
        return None

    @property
    def definitions_mask(self):
        return seen_definitions_mask(self.paths.zsl_split_indices, self.paths.intervals),

    def run(self):
        # TODO(elanmart): This is not atomic :(

        csr_to_tfidf(input_paths=self.to_fit(),
                     input_masks=self.input_masks(),
                     transform_paths=self.to_transform(),
                     output_paths=self.output_paths())

    def output(self):
        return local_targets(
            self.output_paths()
        )


class TfidfJointTrain(TfIdfTaskBase):

    def to_fit(self):
        return [
            self.paths.csr_joint.train,
        ]

    def to_transform(self):
        return self.paths.csr_joint

    def output_paths(self):
        return self.paths.tfidf_j_train


class TfidfJointBoth(TfIdfTaskBase):

    def to_fit(self):
        return [
            self.paths.csr_joint.definitions,
            self.paths.csr_joint.train,
        ]

    def input_masks(self):
        return [
            self.definitions_mask,
            None,
        ]

    def to_transform(self):
        return self.paths.csr_joint

    def output_paths(self):
        return self.paths.tfidf_j_both


class TfidfDisjointTrain(TfIdfTaskBase):

    def to_fit(self):
        return [
            self.paths.csr_disjoint.train,
        ]

    def to_transform(self):
        definitions, *docs = self.paths.csr_disjoint
        return docs

    def output_paths(self):
        definitions, *docs = self.paths.tfidf_dj
        return docs


class TfidfDisjointDefs(TfIdfTaskBase):

    def to_fit(self):
        return [
            self.paths.csr_disjoint.definitions,
        ]

    def input_masks(self):
        return [
            self.definitions_mask,
        ]

    def to_transform(self):
        return [
            self.paths.csr_disjoint.definitions,
        ]

    def output_paths(self):
        return [
            self.paths.tfidf_dj.definitions,
        ]


@inherits(TfIdfTaskBase)
class TfIdfTasks(Wrapper, WikiTask):
    tasks = [
        TfidfJointTrain,
        TfidfJointBoth,
        TfidfDisjointTrain,
        TfidfDisjointDefs,
    ]


@requires(DisjointDocTensor)
class FasttextTask(WikiTask):
    """ This will write documents to FastText format.

    Note that this can be used only for standard classification tasks,
    not zero-shot learning setting.

    Params
    ------
        * TensorCls -> Tensor task used to find the indices to transform into FT.
        * VocabCls  -> Vocabulary task to use to transform indices into text.

    Inputs
    ------
        * {X_train, X_dev, X_test} documents in index-tensor format
        * {Y_train, Y_dev, Y_test} labels associated with documents, in CSR format

    Outputs
    -------
        * {train.ft, dev.ft, test.ft} textfiles. Each textfile continas both documents and labels.
    """

    def run(self):
        with self.tmp_output() as paths_out:

            inputs      = self.input()
            input_paths = [item.path for item in inputs]

            index_to_fasttext(
                doc_paths        = input_paths,
                Y_paths          = self.paths.Y_csr_matrices,
                seen_unseen_path = self.paths.zsl_split,
                y_itos_path      = self.paths.label_itos,
                vocab_path       = self.paths.disjoint_vocab_doc,
                output_paths     = paths_out
            )

    def output(self):
        return local_targets([
            self.paths.fasttext.train,
            self.paths.fasttext.dev,
            self.paths.fasttext.test,
        ])


@inherits(JointDocTensor)
class StarSpaceTask(WikiTask):
    """ This will write documents and definitions to StarSpace format.

    This format is terribly inefficient, so brace yourself and prepare
    a lot of disk space.

    This format supports zero-shot learning scenario.

    Inputs
    ------
        * {Definitions, X_train, X_dev, X_test} documents in index-tensor format
        * Vocabulary, used to build those index-tensors
        * {Y_train, Y_dev, Y_test} labels associated with documents, in CSR format

    Outputs
    -------
        * {train.spc, dev.spc, test.spc} textfiles. Each textfile continas
            both documents and label descriptions
    """

    DefTensorTask = JointDefTensor
    DocTensorTask = JointDocTensor
    VocabTask     = JointVocabBuilder

    def run(self):

        (defs_input,
         docs_input,
         vocab_input,
         ys_input) = self.input()

        basedoc, *outputs = self.output()

        with temp_path(basedoc) as path:
            write_starpace_basedoc(path,
                                   vocab_path     = vocab_input.path,
                                   defs_path      = defs_input.path,
                                   intervals_path = self.paths.intervals)

        for x_input, y_input, out in zip(docs_input, ys_input, outputs):
            with temp_path(out) as out_path:
                write_starpace_file(x_path         = x_input.path,
                                    y_path         = y_input.path,
                                    out_path       = out_path,
                                    vocab_path     = vocab_input.path,
                                    defs_path      = defs_input.path,
                                    intervals_path = self.paths.intervals)

    def output(self):
        return local_targets([
            self.paths.starspace.basedoc,
            self.paths.starspace.train,
            self.paths.starspace.dev,
            self.paths.starspace.test
        ])

    def requires(self):
        kw = self.param_kwargs
        return [
            self.DefTensorTask(**kw),
            self.DocTensorTask(**kw),
            self.VocabTask(**kw),
            YTensorTask(**kw),
        ]


@requires(YsVocab)
class DefinitionVectorsTask(WikiTask):
    """ Extract glove vector for each category, using only its name.

    This task has two parameters. The important one is `VEC_FN`, which
    should return a factory function returning `spacy` model which implements
    `vectors`.
    The `VEC_DIM` param is the dimensionality of the chosen vectors.

    Notes
    -----
    In the future we may want to generate vectors in a different fasion,
    perhaps returning a sequence of vectors (one per word in category name),
    or vectors for both name and definition, or ...?

    Inputs
    ------
        * definitions.json: we're gonna get the vectors directly from the text.

    Outputs
    -------
        * numpy array of size num-labels x vec-dim
    """

    VEC_DIM = 300

    def vec_fn(self):
        return vec_large()

    def run(self):
        with self.tmp_output() as name_avg_path:
            name_vectors(itos_path     = self.paths.label_itos,
                         out_path      = name_avg_path,
                         spacy_factory = self.vec_fn,
                         vec_dim       = self.VEC_DIM)

    def output(self):
        return local_targets([
            self.paths.w2v.name_average,
        ])


@inherits(Vocabs)
class RunAllTask(Wrapper, WikiTask):
    tasks = [
        TensorTasks,
        CsrTasks,
        TfIdfTasks,
        FasttextTask,
        StarSpaceTask,
        DefinitionVectorsTask,
    ]
