import os
import shutil
from functools import partial
from pathlib import Path

import luigi
import numpy as np
from luigi.util import requires, inherits

from pyzsl.data.cub import config
from pyzsl.data.cub.config import Config
from pyzsl.data.cub.paths import CubPaths, TmpPaths
from pyzsl.data.cub.src.core import load_xlsa, \
    run_extraction_pipeline, run_parsing_pipeline, run_splitting_pipeline, \
    clean_images
from pyzsl.utils.general import copytree, json_dump, pickle_dump, \
    pickle_load
from pyzsl.utils.luigi import MkdirTask, DownloadTask, CopyTask, BaseTask, \
    UnpackTask, local_targets, Wrapper


class CUBTask(BaseTask):
    """ This is a base Task, which only provides a single attribute: `paths`

     `paths` is of type `pyzsl.data.cub.paths.CubPaths`
     and makes accessing paths to various files a bit more
     convenient.
    """

    path = luigi.Parameter()  # type: str

    @property
    def paths(self) -> CubPaths:
        return CubPaths(self.path, as_str=True)


class DirectoryStructureTask(luigi.WrapperTask, CUBTask):
    """ Build the directory tree for the dataset.

    """

    def requires(self):

        copy_paths  = [
            (config.__file__, self.paths.config),
        ]

        mkdir_paths = [
            self.paths.tmp,
            self.paths.raw,
            self.paths.features,
            self.paths.labels,
            self.paths.indices,
            self.paths.descriptions,
            self.paths.vocabularies,
            self.paths.metadata,
        ]

        ret  = [CopyTask(src=src, dest=dest)
                for src, dest in copy_paths]

        ret += [MkdirTask(dirname=dirname)
                for dirname in mkdir_paths]

        return ret


@inherits(DirectoryStructureTask)
class GetRawDataTask(luigi.WrapperTask, CUBTask):
    """ Downloads the required data (splits, atributes etc).
    TODO: download reed_data from google drive somehow.

    """

    reed_data = luigi.Parameter()  # type: str

    def _original(self):
        url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
        return DownloadTask(url=url, path=self.paths.orig_c)

    def _xlsa(self):
        url = 'https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip'
        return DownloadTask(url=url, path=self.paths.xlsa_c)

    def _reed(self):
        return CopyTask(src=self.reed_data, dest=self.paths.reed_c),

    def _dir(self):
        return DirectoryStructureTask(path=self.path)

    def requires(self):
        return [
            self._dir(),
            self._reed(),
            self._xlsa(),
            self._original(),
        ]


@inherits(GetRawDataTask)
class UnpackDataTask(luigi.WrapperTask, CUBTask):
    """ Untars / Unzips the datasets

    """

    def requires(self):

        tmp = TmpPaths(self.paths.tmp, as_str=True)

        # TODO: the following is a bit magical... Can we do it nicer?
        # create a Luigi Task class that requires the same params as GetRawDataTask
        UnpackCls = requires(GetRawDataTask)(UnpackTask)
        # When we call this "constructor", it will be automatically fed with self.path and other params
        # we inherited from GetRawDataTask
        Unpack = partial(self.clone, cls=UnpackCls)

        return [
            Unpack(
                input_path=self.paths.orig_c,
                output_dir=self.paths.raw,
                output_paths=[self.paths.orig_images]
            ),
            Unpack(
                input_path=self.paths.reed_c,
                output_dir=tmp.reed,
                output_paths=[tmp.reed]
            ),
            Unpack(
                input_path=self.paths.xlsa_c,
                output_dir=tmp.root,
                output_paths=[tmp.xlsa]
            ),
        ]


@requires(UnpackDataTask)
class MoveTask(CUBTask):

    def all_descriptions(self):

        tmp = TmpPaths(self.paths.tmp)

        src = [
            str(tmp.text_c10 / name.name)
            for name in tmp.text_c10.iterdir()
            if name.is_dir()
        ]

        dst = [
            os.path.join(self.paths.reed_text, name.name)
            for name in tmp.text_c10.iterdir()
            if name.is_dir()
        ]

        return src, dst

    def run(self):

        tmp = TmpPaths(self.paths.tmp)
        Path(self.paths.reed_text).mkdir(exist_ok=True)

        # reed descriptions
        d_src, d_dest = self.all_descriptions()
        for s, d in zip(d_src, d_dest):
            copytree(src=s, dst=d, force=True)

        # xlsa17
        src = [
            tmp.att_splits_mat,
            tmp.trainclasses1,
            tmp.valclasses1,
            tmp.testclasses
        ]

        dst = [
            self.paths.att_splits_mat,
            self.paths.trainclasses1,
            self.paths.valclasses1,
            self.paths.testclasses
        ]

        for s, d in zip(src, dst):
            shutil.copyfile(src=s, dst=d)

    def output(self):

        # _, desc = self.all_descriptions()
        # desc    = local_targets(desc)

        xlsa = local_targets([
            self.paths.att_splits_mat,
            self.paths.trainclasses1,
            self.paths.valclasses1,
            self.paths.testclasses
        ])

        return xlsa  # + desc


class ExtractionBaseTask(CUBTask):

    @property
    def xlsa(self):
        return load_xlsa(self.paths.root)

    @property
    def tmp(self):
        return TmpPaths(self.paths.tmp)


@requires(MoveTask)
class DumpXlsaTask(ExtractionBaseTask):
    def run(self):
        metadata_xlsa, attrs_xlsa, c2i_xlsa = self.xlsa

        itos_dict = {idx: cls for cls, idx in c2i_xlsa.items()}
        itos_list = [itos_dict[i] for i in range(len(itos_dict))]

        np.save(self.paths.attrs, attrs_xlsa)
        json_dump(c2i_xlsa,  self.paths.label_stoi)
        json_dump(itos_list, self.paths.label_itos)

    def output(self):
        return local_targets([
            self.paths.attrs,
            self.paths.label_stoi,
            self.paths.label_itos,
        ])


@requires(DumpXlsaTask)
class ExtractFeaturesTask(ExtractionBaseTask):
    def run(self):

        data  = self.paths.root
        imgs  = self.paths.orig_images
        model = Config.model_loader(Config.device)
        size  = Config.image_size
        norm  = Config.normalize
        clean_images(imgs)

        X, Y, meta = run_extraction_pipeline(data_root=data, img_root=imgs,
                                             model=model, size=size, normalize=norm,
                                             device=Config.device)

        # we save to a temporary location, as splitting will take place later on.
        np.save(self.tmp.X, X)
        np.save(self.tmp.Y, Y)
        pickle_dump(meta, self.tmp.extraction_meta)

    def output(self):
        return local_targets([
            str(self.tmp.X),
            str(self.tmp.Y),
            str(self.tmp.extraction_meta),
        ])


@requires(ExtractFeaturesTask)
class ParseTask(ExtractionBaseTask):
    def run(self):

        data = self.paths.root
        meta = pickle_load(self.tmp.extraction_meta)
        (_,
         filenames) = meta

        (chars,
         words,
         vocab_clvl,
         vocab_wlvl) = run_parsing_pipeline(filenames=filenames, data_root=data)

        np.save(self.paths.char_lvl, chars.long().numpy())
        np.save(self.paths.word_lvl, words.long().numpy())

        json_dump(vocab_clvl, self.paths.char_vocab)
        json_dump(vocab_wlvl, self.paths.word_vocab)
        json_dump(filenames,  self.paths.filenames)

    def output(self):
        return local_targets([
            self.paths.char_lvl,
            self.paths.word_lvl,
            self.paths.char_vocab,
            self.paths.word_vocab,
            self.paths.filenames,
        ])


@requires(ParseTask)
class SplitTask(ExtractionBaseTask):
    def run(self):
        meta, _, c2i_xlsa = self.xlsa
        X = np.load(self.tmp.X)
        Y = np.load(self.tmp.Y)

        # TODO: why is TypeCheck failing here for meta?
        # noinspection PyTypeChecker
        Xs, Ys, IDs, masks = run_splitting_pipeline(X=X, Y=Y,
                                                    meta=meta, label_stoi=c2i_xlsa,
                                                    train_classes=self.paths.trainclasses1,
                                                    dev_classes=self.paths.valclasses1,
                                                    test_classes=self.paths.testclasses)
        
        X_train, X_dev, X_test = Xs
        Y_train, Y_dev, Y_test = Ys
        IDs_train, IDs_dev, IDs_test = IDs
        seen, unseen = masks

        np.save(self.paths.resnet_features.train, X_train)
        np.save(self.paths.label_arrays.train,    Y_train)
        np.save(self.paths.index_arrays.train,    IDs_train)
        
        np.save(self.paths.resnet_features.dev, X_dev)
        np.save(self.paths.label_arrays.dev,    Y_dev)
        np.save(self.paths.index_arrays.dev,    IDs_dev)
        
        np.save(self.paths.resnet_features.test, X_test)
        np.save(self.paths.label_arrays.test,    Y_test)
        np.save(self.paths.index_arrays.test,    IDs_test)

        np.savez(self.paths.testset_mask, seen=seen, unseen=unseen)

    def output(self):
        return local_targets([
            self.paths.resnet_features.train,
            self.paths.label_arrays.train,
            self.paths.index_arrays.train,

            self.paths.resnet_features.dev,
            self.paths.label_arrays.dev,
            self.paths.index_arrays.dev,

            self.paths.resnet_features.test,
            self.paths.label_arrays.test,
            self.paths.index_arrays.test,

            self.paths.testset_mask,
        ])


@inherits(GetRawDataTask)
class RunAllTask(Wrapper, CUBTask):
    tasks = [
        DumpXlsaTask,
        ExtractFeaturesTask,
        ParseTask,
        SplitTask,
    ]
