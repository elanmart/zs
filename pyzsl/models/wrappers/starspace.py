import logging
import os
from signal import SIGTERM
from subprocess import CalledProcessError

import numpy as np
import pexpect

from pyzsl.utils.general import CommandRunner, named_tempfile, mktemp, \
    maybe_tqdm_open
from .base import BaseWrapper


class TrainModes:
    tagspace = 0


class FileFormats:
    fasttext     = 'fastText'
    descritpions = 'labelDoc'


class Executables:
    starspace     = 'starspace'
    query_predict = 'query_predict'


def _parse_results(line):
    predictions = line.split()
    predictions = [p.split(':') for p in predictions]
    predictions = [(int(idx), float(score))
                   for idx, score in predictions]

    (indices,
     values) = zip(*predictions)

    return (np.array(indices),
            np.array(values))


class StarSpace(BaseWrapper):
    def __init__(
            self,
            directory,
            minCount        = 1,
            minCountLabel   = 1,
            ngrams          = 1,
            bucket          = 200000,
            label           = '__label__',
            initModel       ='',
            trainMode       =TrainModes.tagspace,
            fileFormat      = FileFormats.descritpions,
            saveEveryEpoch  = False,
            saveTempModel   = False,
            lr              = 0.01,
            dim             = 100,
            epoch           = 5,
            maxTrainTime    = 8640000,
            negSearchLimit  = 50,
            maxNegSamples   = 10,
            loss            = 'hinge',
            margin          = 0.05,
            similarity      = 'cosine',
            p               = 0.5,
            adagrad         = 1,
            shareEmb        = 1,
            ws              = 5,
            dropoutLHS      = 0,
            dropoutRHS      = 0,
            initRandSd      = 0.001,
            trainWord       = 0,
            wordWeight      = 0.5,
            normalizeText   = False,
            useWeight       = False,
            verbose         = 0,
            debug           = False,
            thread          = 10,
    ):
        """

        Parameters
        ----------
        directory : str
            Path to the root of cloned StarSapce repository.
        minCount :
            minimal number of word occurences [1]
        minCountLabel :
            minimal number of label occurences [1]
        ngrams :
            max length of word ngram [1]
        bucket :
            number of buckets [2000000]
        label :
            labels prefix [__label__]. See file format section.

        initModel :
            if not empty, it loads a previously trained model in -initModel and carry on training.
        trainMode :
            takes value in [0, 1, 2, 3, 4, 5], see Training Mode Section. [0]
        fileFormat :
            currently support 'fastText' and 'labelDoc', see File Format Section. [fastText]
        saveEveryEpoch :
            save intermediate models after each epoch [false]
        saveTempModel :
            save intermediate models after each epoch with an unique name including epoch number [false]
        lr :
            learning rate [0.01]
        dim :
            size of embedding vectors [100]
        epoch :
            number of epochs [5]
        maxTrainTime :
            max train time (secs) [8640000]
        negSearchLimit :
            number of negatives sampled [50]
        maxNegSamples :
            max number of negatives in a batch update [10]
        loss :
            loss function {hinge, softmax} [hinge]
        margin :
            margin parameter in hinge loss. It's only effective if hinge loss is used. [0.05]
        similarity :
            takes value in [cosine, dot]. Whether to use cosine or dot product as
            similarity function in  hinge loss. It's only effective if hinge loss is used. [cosine]

        p :
            normalization parameter: we normalize sum of embeddings by
            deviding Size^p, when p=1, it's equivalent to taking average of
            embeddings; when p=0, it's equivalent to taking sum of embeddings. [0.5]
        adagrad :
            whether to use adagrad in training [1]
        shareEmb :
            whether to use the same embedding matrix for LHS and RHS. [1]
        ws :
            only used in trainMode 5, the size of the context window for word level training. [5]
        dropoutLHS :
            dropout probability for LHS features. [0]
        dropoutRHS :
            dropout probability for RHS features. [0]
        initRandSd :
            initial values of embeddings are randomly generated
            from normal distribution with mean=0, standard deviation=initRandSd. [0.001]
        trainWord :
            whether to train word level together with other tasks (for multi-tasking). [0]
        wordWeight :
            if trainWord is true, wordWeight specifies example weight for word level training examples. [0.5]

        normalizeText :
            whether to run basic text preprocess for input files [0]
        useWeight :
            whether input file contains weights [0]
        verbose :
            verbosity level [0]
        debug :
            whether it's in debug mode [0]
        thread :
            number of threads [10]
        """

        super().__init__()

        def _is_param(name):
            return (not name.startswith('_')) and \
                   (name not in {
                       'self',
                       'directory'
                   })

        self.logger     = logging.getLogger(self.__class__.__name__)

        self.directory  = directory
        self.core_exec  = os.path.join(directory, Executables.starspace)
        self.query_exec = os.path.join(directory, Executables.query_predict)

        self.model_       = None
        self.predictions_ = {}
        self.params       = {
            k: v
            for k, v in locals().items()
            if _is_param(k)
        }

        for k, v in self.params.items():
            self.__setattr__(k, v)

    def _make_params(self, kwargs_only=False, **kwargs) -> str:
        param_dict = kwargs if kwargs_only else {**self.params, **kwargs}
        params     = []

        for k, v in param_dict.items():
            if v == '':
                continue

            if isinstance(v, bool):
                v = int(v)

            params.append(f'-{k} {v}')

        return ' '.join(params)

    def fit(self, path):
        model_path = named_tempfile()

        params  = self._make_params(trainFile=path, model=model_path)
        cmd     = f'{self.core_exec} train {params}'
        runner  = CommandRunner(cmd)

        self.logger.info(f'Running starspace with command {cmd}')

        try:
            runner.run(raise_errors=True)

        except CalledProcessError:
            self.logger.error(f"StarSpace failed. Here's the stderr:\n"
                              f"{runner.get_boxed_output(stdout=True, stderr=True)}")

        else:
            self.logger.info(f'StarSpace exited with retcode 0 :)')
            self.model_ = model_path

    def _query_iterator(self, path, k, basedoc, progbar):
        prompt = 'Enter some text: '

        cmd  = f'{self.query_exec} {self.model_} {k} {basedoc}'
        proc = pexpect.spawn(cmd)
        _    = proc.expect(prompt)

        try:
            with maybe_tqdm_open(path, flag=progbar) as p:
                for line in p:

                    try:
                        doc, _ = line.split('\t', 1)
                    except ValueError:
                        doc = line.rstrip()

                    proc.sendline(doc)
                    proc.expect(prompt)

                    (_doc,
                     response,
                     _empty) = proc.before.decode().split('\n')

                    yield response.strip()

        finally:
            proc.kill(SIGTERM)

    def _file_iterator(self, path, basedoc, k, progbar):

        with mktemp() as preds_path:

            params = self._make_params(
                kwargs_only    = True,
                testFile       = path,
                model          = self.model_,
                K              = k,
                basedoc        = basedoc,
                predictionFile = preds_path,
                thread         = self.thread,
            )

            cmd    = f'{self.core_exec} test {params}'
            runner = CommandRunner(cmd).run(raise_errors=True)

            with maybe_tqdm_open(preds_path, flag=progbar) as f:
                for line in f:
                    yield line.strip()

    def _iterator(self, path, k, basedoc, online, progbar):
        if online:
            return self._query_iterator(path=path, k=k, basedoc=basedoc, progbar=progbar)
        else:
            return self._file_iterator(path=path, k=k, basedoc=basedoc, progbar=progbar)

    def predict_batches(self, path, online=False, basedoc='', k=1, batchsize=1, progbar=False):

        labels = np.zeros((batchsize, k), dtype=np.int32)
        scores = np.zeros((batchsize, k), dtype=np.float32)
        index  = 0

        for row in self._iterator(path=path, basedoc=basedoc, k=k, online=online, progbar=progbar):
            (labels[index, :],
             scores[index, :]) = _parse_results(row)

            index += 1

            if (index % batchsize) == 0:
                yield labels, scores

                labels = np.zeros((batchsize, k), dtype=np.int32)
                scores = np.zeros((batchsize, k), dtype=np.float32)
                index  = 0

        if index > 0:
            yield labels[:index, :], scores[:index, :]

    def predict(self, path, k, basedoc, online=False, progbar=False):
            labels = []
            scores = []

            for row in self._iterator(path=path, basedoc=basedoc, k=k, online=online, progbar=progbar):
                l, s = _parse_results(row)

                labels.append(l)
                scores.append(s)

            return np.stack(labels), np.stack(scores)

    def predict_scores(self, *args, **kwargs):
        raise NotImplementedError(':(')

    def predict_topk(self, *args, **kwargs):
        raise NotImplementedError(':(')

    def predict_ranks(self, *args, **kwargs):
        raise NotImplementedError(':(')

    def __del__(self):
        if (self.model_ is not None) and (os.path.exists(self.model_)):
            os.remove(self.model_)
