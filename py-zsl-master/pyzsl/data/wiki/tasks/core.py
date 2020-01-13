""" Contains tasks from fetching the data up to running WikiExtractor on it.
"""
import os

import luigi
from luigi.contrib.external_program import ExternalProgramTask
from luigi.util import requires, inherits

from pyzsl.data.wiki import config
from pyzsl.data.wiki.config import Config
from pyzsl.data.wiki.paths import WikiPaths
from pyzsl.data.wiki.src.bash import run_wikiextractor
from pyzsl.utils.general import extract_from_tarfile
from pyzsl.utils.luigi import MkdirTask, DownloadTask, S3Task, CopyTask, \
    BaseTask


class WikiTask(BaseTask):
    """ This is a base Task, which only provides a single attribute: `paths`

     `paths` is of type `pyzsl.data.wiki.paths.WikiPaths`
     and makes accessing paths to various files a bit more
     convenient.
    """

    path = luigi.Parameter()  # type: str

    @property
    def paths(self) -> WikiPaths:
        return WikiPaths(self.path, as_str=True)


class DirectoryStructureTask(luigi.WrapperTask, WikiTask):
    """ Create a basic directory structure for this dataset.

    Notes
    -----
    path: str
        A path to a base directory where this dataset should reside

    """

    def requires(self):

        copy_paths  = [
            (config.__file__, self.paths.config),
        ]

        mkdir_paths = [
            self.paths.tmp,
            self.paths.raw,
            self.paths.processed,
            self.paths.vocabularies,
            self.paths.metadata,
            self.paths.labels,
            self.paths.json_files,
            self.paths.index_arrays,
            self.paths.csr_matrices,
            self.paths.tfidf_matrices,
            self.paths.misc_formats,
            self.paths.word_vectors,
        ]

        ret  = [CopyTask(src=src, dest=dest)
                for src, dest in copy_paths]

        ret += [MkdirTask(dirname=dirname)
                for dirname in mkdir_paths]

        return ret


@inherits(DirectoryStructureTask)
class GetRawDataTask(luigi.WrapperTask, WikiTask):
    """ Downloads raw data from the web and creates directory structure.

    Downloaded data includes
    * raw wikipedia dump
    * raw dictionary from http://www.cl.cam.ac.uk/~fh295/

    Notes
    -----
    date: str
        Date identifier for the wikipedia dump to use.
        defualt: Config.Source.date
    wiki: str
        Name of the wikipedia dump to download. Currently {'simlpewiki', 'enwiki'} are supported.
        default: Config.Source.wiki
    site: NotImplemented
        Website to fetch the dump from. In the future we may want to host one particular dump ourselves
    """

    date = luigi.Parameter(Config.Source.date)  # type: str
    wiki = luigi.Parameter(Config.Source.wiki)  # type: str
    site = luigi.Parameter(None)

    def _wiki(self):
        site   = 'https://dumps.wikimedia.org'
        suffix = 'pages-articles.xml.bz2'
        url    = f'{site}/{self.wiki}/{self.date}/{self.wiki}-{self.date}-{suffix}'

        return DownloadTask(url=url, path=self.paths.dump_xml_bz2)

    def _dict(self):
        url  = 'http://www.cl.cam.ac.uk/~fh295/training_data.tgz'
        md5  = '089e0631fcbd4f1c2e23aecec44cd30c'
        path = self.paths.dictionary_tgz

        return DownloadTask(url=url, path=path, md5=md5)

    def _dir(self):
        return DirectoryStructureTask(path=self.path)

    def requires(self):
        return [
            self._dir(),
            self._wiki(),
            self._dict(),
        ]


@requires(GetRawDataTask)
class DecompressWikiTask(ExternalProgramTask, WikiTask):
    """ Decompress the downloaded wikipedia dump.
    """

    def program_args(self):
        # bzip2 takes care of not creating partial results, so its :)
        cmd = f'bzip2 -dk {self.paths.dump_xml_bz2}'

        return cmd.split()

    def output(self):
        return luigi.LocalTarget(self.paths.dump_xml)


@requires(DecompressWikiTask)
class DecompressDictTask(WikiTask):
    """ Decompress the downloaded dictionary.

    Checks the md5 sum of the extracted thing, just to be extra sure
    we're all good, as it's a python object, so we don't want to
    read the wrong thing...
    """

    def run(self):
        member = 'training_data/training_data.pkl'
        md5    = '516b7d06ae60e7a446c8f8ed6e7a035c'

        with self.tmp_output() as tmp_path:
            extract_from_tarfile(self.paths.dictionary_tgz, member, tmp_path, md5=md5)

    def output(self):
        return luigi.LocalTarget(self.paths.dictionary)


@requires(DecompressDictTask)
class ExtarctTask(WikiTask):
    """ Run WikiExtractor on the decompressed data. Perform optional preprocessing.

    The preprocessing is kind of a legacy thing -- there are now multiple places
    in the code where preprocessing can be applied, this is perhaps not the best
    one...

    Notes
    -----
    lowercase: bool
        if True, lowercase all text
    tokenize: bool
        if True, use spacy Normalizer on all text fields
    replace_num: bool
        if True, replace all numbers with NUM token
    lemmatize: bool
        if True, use spacy lemmatizer on all text fields
    """

    lowercase       = luigi.BoolParameter(Config.Extraction.lowercase)     # type: bool
    tokenize        = luigi.BoolParameter(Config.Extraction.tokenize)      # type: bool
    replace_num     = luigi.BoolParameter(Config.Extraction.replace_num)   # type: bool
    lemmatize       = luigi.BoolParameter(Config.Extraction.lemmatize)     # type: bool

    def _run_wikiextractor(self):
        with self.tmp_output() as output:
            run_wikiextractor(input=self.paths.dump_xml, output=output,
                              tokenize=self.tokenize, lowercase=self.lowercase,
                              replace_num=self.replace_num, lemmatize=self.lemmatize)

    def run(self):
        # TODO(elan): remove this after spacy download is added to setup.py
        self._run_wikiextractor()

    def output(self):
        return luigi.LocalTarget(self.paths.dump_json)


@requires(ExtarctTask)
class SendToS3Task(S3Task, WikiTask):
    """ Send the extracted data to S3. Legacy, not tested recently

    Notes
    -----
    bucket: str
        S3 bucket to put the data in.
    """

    bucket = luigi.Parameter()  # type: str

    def s3_path(self):
        dirname = os.path.basename(self.paths.root)
        fname   = os.path.basename(self.input().path)
        return f'{dirname}/{fname}'

    def fs_path(self):
        return self.input().path
