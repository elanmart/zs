import gc
import logging

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from pyzsl.data.wiki.src.vocabs import Vocab
from pyzsl.utils.general import dill_load, numpy_dump, numpy_load, \
    json_load, scipy_dump, scipy_load, scipy_load_optimized, dill_dump, \
    load_zsl_split


def tensor_to_csr(path, n_cols=None):
    X = np.load(path)
    n_cols = n_cols or (X.max() + 1)

    indptr     = np.zeros((X.shape[0] + 1,), dtype=np.int64)
    indptr[1:] = (X > 0).sum(1)
    indptr     = np.cumsum(indptr)

    indices = X.reshape(-1)
    indices = indices[indices > 0]

    data = np.ones_like(indices, dtype=np.float32)

    csr = sp.csr_matrix((data, indices, indptr), shape=(X.shape[0], n_cols))
    csr.check_format(full_check=True)

    return csr


def csr_to_tfidf(input_paths, transform_paths, output_paths, input_masks=None):
    logger = logging.getLogger(csr_to_tfidf.__name__)

    X = [scipy_load_optimized(p) for p in input_paths]

    if input_masks is not None:
        X = [x[mask] if (mask is not None) else x
             for x, mask in zip(X, input_masks)]

    X = sp.vstack(X)
    gc.collect()

    tfidf = TfidfTransformer()
    tfidf.fit(X)

    del X
    gc.collect()

    for input, output in zip(transform_paths, output_paths):
        logger.debug(f'Loading sparse csr matrix from {input}')
        X = scipy_load_optimized(input)

        logger.debug(f'Copying  {X.__repr__()} ')
        X = X.copy()

        dill_dump((X, tfidf), '/tmp/foo.dill')

        logger.debug(f'Transforming into tf-idf format.')
        X = tfidf.transform(X)

        logger.debug(f'Storing matrix {X.__repr__()} at {output}')
        scipy_dump(X, output)

        logger.debug('done')


def index_to_fasttext(doc_paths, Y_paths,
                      vocab_path, y_itos_path, seen_unseen_path,
                      output_paths):

    vocab   = dill_load(vocab_path)
    y_itos  = json_load(y_itos_path)
    seen, _ = load_zsl_split(seen_unseen_path)

    for path_in, path_y, path_out in zip(doc_paths, Y_paths, output_paths):

        X = numpy_load(path_in)
        Y = scipy_load(path_y)

        with open(path_out, 'w') as f_out:
            desc = f'fasttext: == {path_out} =='

            for x, y in tqdm(zip(X, Y), desc=desc, total=X.shape[0]):

                words = vocab.from_indices(x[x > 0])

                labels = [
                    y_itos[idx]
                    for idx in y.indices
                ]

                labels = [
                    f"__label__{l.replace(' ', '_')}"
                    for l in labels
                    if l in seen
                ]

                if len(labels) == 0:
                    continue

                labels = ' '.join(labels)
                print(f'{words} {labels}', file=f_out)


def write_starpace_basedoc(path, vocab_path, defs_path, intervals_path):
    with open(path, 'w') as f:

        vocab     = dill_load(vocab_path)  # type: Vocab
        defs      = numpy_load(defs_path)
        intervals = numpy_load(intervals_path)

        n = intervals.shape[0]
        for idx in range(n):
            start, stop = intervals[idx]
            definition  = defs[start]
            words       = vocab.from_indices(definition)

            print(words, file=f)


def write_starpace_file(x_path, y_path, out_path,
                        vocab_path, defs_path, intervals_path):

    X     = numpy_load(x_path)
    Y     = scipy_load(y_path)

    vocab     = dill_load(vocab_path)  # type: Vocab
    defs      = numpy_load(defs_path)
    intervals = numpy_load(intervals_path)

    with open(out_path, 'w') as f_out:

        for row_x, labels in zip(X, Y):
            row_x   = row_x[row_x != 0]
            words_x = vocab.from_indices(row_x)

            line = words_x

            for label in labels.indices:
                start, stop = intervals[label]
                definition  = defs[start]
                words_y     = vocab.from_indices(definition[definition > 0])

                line += f'\t{words_y}'

            print(line, file=f_out)


def name_vectors(itos_path, out_path, spacy_factory, vec_dim):
    nlp  = spacy_factory()
    itos = json_load(itos_path)
    ret  = np.zeros((len(itos), vec_dim), dtype=np.float32)

    for idx, name in enumerate(itos):
        ret[idx, :] = nlp(name).vector

    numpy_dump(ret, out_path)
