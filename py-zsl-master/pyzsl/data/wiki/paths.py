from pathlib import Path

from pyzsl.utils.general import PathBase, as_named_tuple


class DataStub:
    definitions = train = dev = test = ''


class StarspaceStub:
    basedoc = train = dev = test = ''


class WikiPaths(PathBase):
    # root
    root = Path('')

    # config dump
    config = root / 'config.py'

    # directories
    tmp            = Path('tmp')             # files that can be safely removed after generation process exits.
    raw            = Path('raw')             # raw downladed data
    processed      = Path('processed')       # data after initial preprocessing
    vocabularies   = Path('vocabularies')    # all objects mapping abstract indices to human-readable words
    metadata       = Path('metadata')        # any additional objects describing the data
    labels         = Path('labels')          # where labels associated with each example will be stored
    json_files     = Path('json-files')      # final data in json format
    index_arrays   = Path('index-arrays')    # final data in index-array format
    csr_matrices   = Path('csr-matrices')    # final data in csr format
    tfidf_matrices = Path('tfidf-matrices')  # final data in tf-idf format
    misc_formats   = Path('misc-formats')    # final data in misc formats
    word_vectors   = Path('word-vectors')    # data as word-vectors

    # downloaded wikipedia dump
    dump_xml_bz2 = raw / 'raw-wiki-dump.xml.bz2'
    dump_xml     = raw / dump_xml_bz2.name.replace('.bz2', '')
    dump_json    = processed / 'wiki-dump-extracted.json'

    # dictionary of label definitions
    dictionary_tgz  = raw       / 'training_data.tgz'
    dictionary      = processed / 'training_data.pkl'

    # zsl splits --> assignment of labels to "seen" vs "unseen" groups
    tmp_zsl_split     = tmp      / 'zsl-split.json'
    zsl_split         = metadata / 'zsl-split.json'
    zsl_split_indices = metadata / 'zsl-split.npz'

    # definitions metadata
    label_stoi        = metadata / 'label-stoi.json'       # mapping label name to label index
    label_itos        = metadata / 'label-itos.json'       # mapping label index to label name
    definition_labels = metadata / 'defintion_labels.npy'  # size: (num_definitions, ), * (see bottom of the file)
    intervals         = metadata / 'intervals.npy'         # size: (num_labels, 2), ** (see bottom of the file)

    # serialized vocablary objects
    joint_vocab    = vocabularies / 'joint-vocab.dill'
    disjoint_vocab_def = vocabularies / 'disjoint-vocab-def.dill'
    disjoint_vocab_doc = vocabularies / 'disjoint-vocab-doc.dill'

    # temporary
    tmp_data = as_named_tuple(
        'tmp_data',
        definitions = tmp / 'definitions.json',
        train       = tmp / 'train.json',
        dev         = tmp / 'dev.json',
        test        = tmp / 'test.json'
    )  # type: DataStub

    # ready-to-use, human readable documents in json format
    json_data = as_named_tuple(
        'json_data',
        definitions = json_files / 'definitions.json',
        train       = json_files / 'train.json',
        dev         = json_files / 'dev.json',
        test        = json_files / 'test.json',
    )  # type: DataStub

    # labels as CSR matrices
    Y_csr_matrices = as_named_tuple(
        'Y_csr_matrices',
        train = labels / 'y_train.csr.npz',
        dev   = labels / 'y_dev.csr.npz',
        test  = labels / 'y_test.csr.npz'
    )  # type: DataStub

    # data as index arrays built from joint vocab
    index_data_joint = as_named_tuple(
        'joint_index_data',
        definitions = index_arrays / 'joint_definitions.npy',
        train       = index_arrays / 'joint_train.npy',
        dev         = index_arrays / 'joint_dev.npy',
        test        = index_arrays / 'joint_test.npy'
    )  # type: DataStub

    # data as index arrays built from disjoint vocab
    index_data_disjoint = as_named_tuple(
        'disjoint_index_data',
        definitions = index_arrays / 'disjoint_definitions.npy',
        train       = index_arrays / 'disjoint_train.npy',
        dev         = index_arrays / 'disjoint_dev.npy',
        test        = index_arrays / 'disjoint_test.npy'
    )  # type: DataStub

    # data as CSR matrices built from joint vocab
    csr_joint = as_named_tuple(
        'joint_csr',
        definitions = csr_matrices / 'joint_definitions.csr.npz',
        train       = csr_matrices / 'joint_train.csr.npz',
        dev         = csr_matrices / 'joint_dev.csr.npz',
        test        = csr_matrices / 'joint_test.csr.npz',
    )  # type: DataStub

    # data as CSR matrices built from disjoint vocab
    csr_disjoint = as_named_tuple(
        'disjoint_csr',
        definitions = csr_matrices / 'disjoint_definitions.csr.npz',
        train       = csr_matrices / 'disjoint_train.csr.npz',
        dev         = csr_matrices / 'disjoint_dev.csr.npz',
        test        = csr_matrices / 'disjoint_test.csr.npz',
    )  # type: DataStub

    # data as CSR matrices of tf-idf values, built from joint vocab and train documents only
    tfidf_j_train = as_named_tuple(
        'train_j_tfidf',
        definitions  = csr_matrices / 'train-j_definitions.tfidf.csr.npz',
        train        = csr_matrices / 'train-j_train.tfidf.csr.npz',
        dev          = csr_matrices / 'train-j_dev.tfidf.csr.npz',
        test         = csr_matrices / 'train-j_test.tfidf.csr.npz',
    )  # type: DataStub

    # data as CSR matrices of tf-idf values, built from joint vocab and train documents + seen label definitions
    tfidf_j_both = as_named_tuple(
        'both_j_tfidf',
        definitions  = csr_matrices / 'both-j_definitions.tfidf.csr.npz',
        train        = csr_matrices / 'both-j_train.tfidf.csr.npz',
        dev          = csr_matrices / 'both-j_dev.tfidf.csr.npz',
        test         = csr_matrices / 'both-j_test.tfidf.csr.npz',
    )  # type: DataStub

    # data as CSR matrices of tf-idf values, built from disjoint vocab and train documents only
    tfidf_dj = as_named_tuple(
        'dj_tfidf',
        definitions  = csr_matrices / 'train-dj_definitions.tfidf.csr.npz',
        train        = csr_matrices / 'train-dj_train.tfidf.csr.npz',
        dev          = csr_matrices / 'train-dj_dev.tfidf.csr.npz',
        test         = csr_matrices / 'train-dj_test.tfidf.csr.npz',
    )  # type: DataStub

    # data transformed to word vectors
    w2v = as_named_tuple(
        'w2v',
        name_average = word_vectors / 'name-avg-wv.npy'
    )

    # data for fasttext training
    fasttext = as_named_tuple(
        'fasttext',
        train = misc_formats / 'train.ft',
        dev   = misc_formats / 'dev.ft',
        test  = misc_formats / 'test.ft',
    )  # type: DataStub

    # data for starspace training
    starspace = as_named_tuple(
        'starspace',
        basedoc = misc_formats / 'basedoc.spc',
        train   = misc_formats / 'train.spc',
        dev     = misc_formats / 'dev.spc',
        test    = misc_formats / 'test.spc',
    )  # type: StarspaceStub

# *
# NB: we have multiple definitions per label
#     so for each definition we need to remember which label it describes (`definition_labels` array)
#     e.g. [0, 0, 0, 1, 1, 2, ...] means first three definitions are of label 0, and so on.

# **
# NB: we may want to quickly access definitions of label index `i`.
#     This array describes exactly what indices interval gives you definitions for label `i`
#     for the example above, we'd have intervals = [(0, 3), (3, 5), (5, ...), ...]
