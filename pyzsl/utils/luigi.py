import os
import shutil
from contextlib import contextmanager
from typing import List, Iterable
from urllib.request import urlretrieve
from tqdm import tqdm
import boto3
import luigi
from luigi.contrib import s3
from luigi.contrib.external_program import ExternalProgramTask
from luigi.util import inherits

from pyzsl.utils.general import file_md5, get_contexts, dprint


def local_targets(paths):
    return [luigi.LocalTarget(p) for p in paths]


def wrapper(name, bases, inherit_from, tasks):
    def requires(self):
        return [
            cls(**self.param_kwargs)
            for cls in tasks
        ]

    cls = type(name, (luigi.WrapperTask, *bases), {'requires': requires})
    cls = inherits(inherit_from)(cls)

    return cls


@contextmanager
def temp_path(output: luigi.LocalTarget):
    with output.temporary_path() as pth:
        yield pth


@contextmanager
def temp_file(output: luigi.LocalTarget, *args, **kwargs):
    with output.temporary_path() as pth:
        with open(pth, *args, **kwargs) as f:
            yield f


class BaseTask(luigi.Task):
    def tmp_output(self):
        """
        The following two snippets are equivalent

        >>> with self.output().temporary_path() as output_path:
        ...     with open(output_path, 'w') as f:
        ...         pass

        >>> with self.tmp_output() as output_path:
        ...     with open(output_path, 'w') as f:
        ...         pass

        And the following two snippets are also equivalent:

        >>> output_1, output_2 = self.output()
        >>> with output_1.temporary_path() as output_path_1, output_2.temporary_path() as output_path_2:
        ...     with open(output_path_1, 'w') as f1, open(output_path_2, 'w') as f2:
        ...         pass

        >>> with self.tmp_output() as (output_path_1, output_path_2):
        ...     with open(output_path_1, 'w') as f1, open(output_path_2, 'w') as f2:
        ...         pass
        """

        output = self.output()
        if not isinstance(output, Iterable):
            output = [output]

        managers = [o.temporary_path() for o in output]
        contexts = get_contexts(managers)

        return contexts


class Wrapper(luigi.WrapperTask):
    tasks = NotImplemented

    def requires(self):
        return [
            cls(**self.param_kwargs)
            for cls in self.tasks
        ]


class MkdirTask(luigi.Task):
    """ Creates directory if it doesn't exist """

    dirname = luigi.Parameter()  # type: str

    def run(self):
        os.makedirs(self.dirname, exist_ok=True)

    def output(self):
        return luigi.LocalTarget(self.dirname)


class S3Task(luigi.Task):
    """ Puts dataset to an S3 Bucket. It seemed easier to write these few lines than to
    use luigi's built-in AWS support. """

    def bucket(self):
        raise NotImplementedError

    def fs_path(self):
        raise NotImplementedError

    def s3_path(self):
        raise NotImplementedError

    def run(self):
        bucket = boto3.resource('s3').Bucket(self.bucket())
        bucket.upload_file(
            self.fs_path(),
            self.s3_path()
        )

    def output(self):
        return s3.S3Target(f's3://{self.bucket()}/{self.s3_path()}')


class DownloadTask(BaseTask):
    """ Downloads a file from `url` and stores it under `path` """

    url       = luigi.Parameter()    # type: str
    path      = luigi.Parameter()    # type: str
    md5       = luigi.Parameter('')  # type: str
    _tmp_path = NotImplemented

    def requires(self):
        return MkdirTask(os.path.dirname(self.path))

    def run(self):
        pbar = None
        def print_progress(num_blocks, b_size, total_size):
            nonlocal pbar
            if not pbar:
                pbar = tqdm(total=total_size, unit_scale=True)
            else:
                pbar.update(b_size)

        with self.tmp_output() as tmp_path:
            urlretrieve(self.url, tmp_path, print_progress)

            if self.md5 != '':
                file_md5(tmp_path, self.md5)
        del pbar            

    def output(self):
        return luigi.LocalTarget(self.path)


class CopyTask(luigi.Task):
    src = luigi.Parameter()  # type: str
    dest = luigi.Parameter()  # type: str

    def requires(self):
        return MkdirTask(dirname=os.path.dirname(self.dest))

    def run(self):
        with self.output().temporary_path() as dest:
            shutil.copyfile(self.src, dest)

    def output(self):
        return luigi.LocalTarget(self.dest)


class UnpackTask(ExternalProgramTask):
    """ Unpacks a zip or a tarball to output_path """

    input_path   = luigi.Parameter()          # type: str
    output_dir   = luigi.Parameter()          # type: str
    output_paths = luigi.ListParameter()      # type: List[str]
    makedir      = luigi.BoolParameter(True)  # type: bool

    def program_args(self):

        if self.makedir:
            os.makedirs(self.output_dir, exist_ok=True)

        if self.input_path.endswith('tar.gz') or self.input_path.endswith(
                'tgz'):
            cmd = f'tar -xvf {self.input_path} --directory {self.output_dir}'

        elif self.input_path.endswith('zip'):
            cmd = f'unzip {self.input_path} -d {self.output_dir}'

        else:
            raise RuntimeError('Archive type not supported')

        return cmd.split()

    def output(self):
        return [
            luigi.LocalTarget(fname)
            for fname in self.output_paths
        ]
