import luigi
from luigi.util import requires
import numpy as np
from pyzsl.data.awa.paths import AWAPaths, TmpPaths
from pyzsl.utils.luigi import Wrapper, MkdirTask, DownloadTask, \
                              UnpackTask, local_targets
from pyzsl.utils.general import json_dump, json_load, \
                                numpy_array_from_text, readlines
from functools import partial


class AWATask(luigi.Task):
    """ This task provides path and tmp_path property to all tasks. """
    path = luigi.Parameter() # type: str
    @property
    def paths(self) -> AWAPaths:
        return AWAPaths(self.path, as_str=True)

    @property
    def tmp_paths(self) -> TmpPaths:
        return TmpPaths(self.paths.tmp, as_str=True)


class DirectoryStructureTask(luigi.WrapperTask, AWATask):
    def requires(self):
        mkdir_paths = [
              self.paths.tmp
            , self.paths.raw
            , self.paths.features
            , self.paths.labels
            , self.paths.indices
            , self.paths.descriptions
            , self.paths.metadata
        ]

        ret = [MkdirTask(dirname=d) for d in mkdir_paths]
        return ret

class GetRawDataTask(luigi.WrapperTask, AWATask):
    def _base_data(self):
        url = "https://cvml.ist.ac.at/AwA2/AwA2-base.zip"
        return DownloadTask(url=url, path=self.paths.base_zip,
                            md5="27998437f72823d8ac314257682b57ca")

    def _img_features(self):
        url = "http://cvml.ist.ac.at/AwA2/AwA2-features.zip"
        return DownloadTask(url=url, path=self.paths.features_zip,
                            md5="b1735c7b8a9044b1d51903b4c9e9fcd5")

    def _xlsa(self):
        url = 'https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip'
        return DownloadTask(url=url, path=self.paths.xlsa_zip)

    def _dir(self):
        return DirectoryStructureTask(path=self.path)

    def requires(self):
        return [
            self._dir(),
            self._base_data(),
            self._img_features(),
            self._xlsa()
        ]

class UnpackDataTask(luigi.WrapperTask, AWATask):
    def requires(self):
        tmp_paths = TmpPaths(self.paths.tmp, as_str=True)

        # To unpack data, we need to have it downloaded. Since we can't force
        # downloading to happen before unpacking using this requires method, we
        # create decorated UnpackTask, that requires GetRawDataTask. Since
        # UnpackCls requires GetRawDataTask, we are sure that data is downloaded
        # before it's used.
        UnpackCls = luigi.util.requires(GetRawDataTask)(UnpackTask)

        # Normally when task '@requires' another task, this call happens behind
        # the scenes. self.clone copies self's params, and calls 'cls' with them
        # We could as well explicitly pass all needed self's params to UnpackCls
        Unpack = partial(self.clone, cls=UnpackCls)

        return [
            Unpack(
                input_path=self.paths.base_zip,
                output_dir=tmp_paths.root,
                output_paths=[tmp_paths.base_dir]
            ),
            Unpack(
                input_path=self.paths.features_zip,
                output_dir=tmp_paths.root,
                output_paths=[tmp_paths.features_dir]
            ),
            Unpack(
                input_path=self.paths.xlsa_zip,
                output_dir=tmp_paths.root,
                output_paths=[tmp_paths.xlsa_dir]
            )
        ]

@requires(UnpackDataTask)
class ProcessAWABaseTask(AWATask):
    def _read_and_clean(self, path_to_file):
        names = readlines(path_to_file) # format: numer\name
        # We ignore numbers, since we perfer 0 based indexing
        names = [line.split("\t")[1] for line in names] # list of names

        idx_to_name = names
        name_to_idx = {name : id for (id, name) in enumerate(names)}

        return (idx_to_name, name_to_idx)

    def map_labels(self):
        (idx_to_label,
         label_to_idx) = self._read_and_clean(self.tmp_paths.classes_txt)

        json_dump(idx_to_label, self.paths.label_itos)
        json_dump(label_to_idx, self.paths.label_stoi)

    def map_attrs(self):
        (idx_to_attrib,
         attrib_to_idx) = self._read_and_clean(self.tmp_paths.attrs_txt)

        json_dump(idx_to_attrib, self.paths.attrs_itos)
        json_dump(attrib_to_idx, self.paths.attrs_stoi)

    def create_attributes_matrix(self):
        bin_mat  = numpy_array_from_text(self.tmp_paths.attrs_matrix_bin, dtype=np.int)
        cont_mat = numpy_array_from_text(self.tmp_paths.attrs_matrix_cont)
        assert bin_mat.size == cont_mat.size
        np.save(self.paths.attrs_bin, bin_mat)
        np.save(self.paths.attrs_cont, cont_mat)

    def run(self):
        self.map_labels()
        self.map_attrs()
        self.create_attributes_matrix()

    def output(self):
        return local_targets([
              self.paths.label_itos
            , self.paths.label_stoi
            , self.paths.attrs_itos
            , self.paths.attrs_stoi
            , self.paths.attrs_bin
            , self.paths.attrs_cont
        ])


@requires(ProcessAWABaseTask)
class CreateTrainTestSplit(AWATask):
    """ Create train/test mask for each element.
        Additionally, create list of train/test/val indices.
    """
    dev_size = luigi.IntParameter(default=500)

    def _create_split(self):
        # turn into 0 based idx -> -1
        labels = numpy_array_from_text(self.tmp_paths.labels, dtype=np.int64) - 1

        # Read split - format: class+name
        trainval_classes = readlines(self.tmp_paths.trainval_classes)
        test_classes = readlines(self.tmp_paths.test_classes)

        label_stoi = json_load(self.paths.label_stoi)
        # Turn class names to class indices
        trainval_class_indexes = [label_stoi[name] for name in trainval_classes]
        test_class_indexes     = [label_stoi[name] for name in test_classes]

        train_mask = np.isin(labels, trainval_class_indexes)
        test_mask  = np.isin(labels, test_class_indexes)
        assert all(train_mask | test_mask)

        np.savez(self.paths.testset_mask, seen=train_mask, unseen=test_mask)
        assert train_mask.ndim == 1
        train_indices = np.nonzero(train_mask)[0]
        dev_indices   = np.random.choice(train_indices, self.dev_size, replace=False)
        test_indices  = np.nonzero(test_mask)[0]
        assert train_indices.ndim == 1
        assert dev_indices.ndim   == 1
        assert test_indices.ndim  == 1
        np.save(self.paths.index_arrays.train, train_indices)
        np.save(self.paths.index_arrays.dev,   dev_indices)
        np.save(self.paths.index_arrays.test,  test_indices)

    def run(self):
        self._create_split()

    def output(self):
        return local_targets([
              self.paths.testset_mask
            , self.paths.index_arrays.train
            , self.paths.index_arrays.dev
            , self.paths.index_arrays.test
        ])


@requires(CreateTrainTestSplit)
class SplitFeaturesAndLabels(AWATask):
    def _split(self):
        features = numpy_array_from_text(self.tmp_paths.features)
        # turn into 0 based idx -> -1
        labels = numpy_array_from_text(self.tmp_paths.labels, dtype=np.int64) - 1

        train_indices  = np.load(self.paths.index_arrays.train)
        dev_indices    = np.load(self.paths.index_arrays.dev)
        test_indices   = np.load(self.paths.index_arrays.test)
        train_labels = labels[train_indices]
        dev_labels   = labels[dev_indices]
        test_labels  = labels[test_indices]
        np.save(self.paths.label_arrays.train, train_labels)
        np.save(self.paths.label_arrays.dev,   dev_labels)
        np.save(self.paths.label_arrays.test,  test_labels)

        train_features = features[train_indices]
        dev_features   = features[dev_indices]
        test_features  = features[test_indices]
        np.save(self.paths.resnet_features.train, train_features)
        np.save(self.paths.resnet_features.dev,   dev_features)
        np.save(self.paths.resnet_features.test,  test_features)

    def run(self):
        self._split()

    def output(self):
        return local_targets([
              self.paths.label_arrays.train
            , self.paths.label_arrays.dev
            , self.paths.label_arrays.test
            , self.paths.resnet_features.train
            , self.paths.resnet_features.dev
            , self.paths.resnet_features.test
        ])

class RunAllTask(Wrapper, AWATask):
    tasks = [SplitFeaturesAndLabels]

class Validate(AWATask):
    def _check_label_mapping(self):
        label_stoi = json_load(self.paths.label_stoi)
        label_itos = json_load(self.paths.label_itos)
        assert label_itos[0]  == "antelope"
        assert label_itos[49] == "dolphin"
        for (idx, name) in enumerate(label_itos):
            assert idx == label_stoi[name]

    def _check_attrs_mapping(self):
        attrs_stoi = json_load(self.paths.attrs_stoi)
        attrs_itos = json_load(self.paths.attrs_itos)
        assert attrs_itos[0]  == 'black'
        assert attrs_itos[84] == 'domestic'
        for (idx, name) in enumerate(attrs_itos):
            assert idx == attrs_stoi[name]

    def _check_indices(self):
        train_indices = np.load(self.paths.index_arrays.train)
        assert (    0 in train_indices)
        assert (20173 in train_indices)
        assert (22400 in train_indices)
        test_indices = np.load(self.paths.index_arrays.test)
        assert (30600 in test_indices)      # first sheep
        assert (32019 in test_indices)      # last sheep
        assert ( 1796 in test_indices)      # first bobcat
        assert ( 2425 in test_indices)      # last bobcat
        assert ( 1622 in test_indices)      # first blue+whale
        assert ( 1046 in test_indices)      # first bat
        dev_indices = np.load(self.paths.index_arrays.dev)
        assert np.intersect1d(train_indices, test_indices).size == 0
        assert np.intersect1d(dev_indices, test_indices).size == 0

    def _check_labels(self):
        train_labels = np.load(self.paths.label_arrays.train)
        assert train_labels[0]  == 0        # antelope, first train class
        assert train_labels[-1] == 37       # zebra, last train class
        test_labels = np.load(self.paths.label_arrays.test)
        assert test_labels[0]  == 29        # bat, first test class
        assert test_labels[-1] == 46        # walrus, last tets class

    def _check_features(self):
        # Quick and dirty check for number similarity
        similar = lambda x, y: abs(x-y) < 1e-4

        train_features = np.load(self.paths.resnet_features.train)
        # first antelope
        assert similar(train_features[0,0],    0.12702841)
        assert similar(train_features[0,-1],   0.40761620)
        # last zebra
        assert similar(train_features[-1, 0],  0.24346060)
        assert similar(train_features[-1, -1], 0.06219054)

        test_features = np.load(self.paths.resnet_features.test)
        # first bat
        assert similar(test_features[0,0],  0.53858560)
        assert similar(test_features[0,-1], 0.06032756)
        # last walrus
        assert similar(test_features[-1,0],  0.55041420)
        assert similar(test_features[-1,-1], 0.56979007)

    def run(self):
        self._check_label_mapping()
        self._check_attrs_mapping()
        self._check_indices()
        self._check_labels()
        self._check_features()
        print("All checks successful")