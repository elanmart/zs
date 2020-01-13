import os
import subprocess
from os.path import join

import luigi
from luigi.util import requires

from pyzsl.utils.luigi import MkdirTask, DownloadTask


class GetDataTask(luigi.WrapperTask):

    dirname     = luigi.Parameter()                              # type: str
    posts_fname = luigi.Parameter('stackoverflow.com-Posts.7z')  # type: str
    tags_fname  = luigi.Parameter('stackoverflow.com-Tags.7z')   # type: str

    def requires(self):
        return [
            MkdirTask(dirname=self.dirname),
            DownloadTask(dirname=self.dirname, name=self.posts_fname),
            DownloadTask(dirname=self.dirname, name=self.tags_fname),
        ]

    def output(self):
        return [luigi.LocalTarget(join(self.dirname, name))
                for name in [self.posts_fname, self.tags_fname]]


@requires(GetDataTask)
class DecompressTask(luigi.Task):

    def _get_paths(self):
        posts, tags = self.input()
        dirname = os.path.dirname(posts.path)

        return dirname, posts.path, tags.path

    def run(self):
        dirname, posts, tags = self._get_paths()

        for path in posts, tags:
            subprocess.check_call(f'7z x {path} -o{dirname}'.split())

    def output(self):
        dirname, *_ = self._get_paths()

        return [luigi.LocalTarget(join(dirname, name))
                for name in ['Posts.xml', 'Tags.xml']]
