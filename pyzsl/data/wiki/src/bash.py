import os
import subprocess

from pyzsl.utils.general import normalize_path


def run_wikiextractor(input: str,
                      output: str,
                      tokenize: bool,
                      lowercase: bool,
                      replace_num: bool,
                      lemmatize: bool):
    """ Run the WikiExtractor script with given options

    Parameters
    ----------
    input:
        Path to wiki dump
    output:
        Path where to extracted file will be stored
    tokenize:
        if True, perform tokenization of the text
    lowercase:
        if True, convert all text to lowercase
    replace_num:
        if True, replace numbers with a special token
    lemmatize:
        if True, use spacy lemmatization on all tokens
    """

    here      = os.path.dirname(normalize_path(__file__))
    extractor = os.path.join(here, 'WikiExtractor.py')

    cmd  = f'python {extractor} '
    cmd += '-o - -q --sections --json --filter_disambig_pages -ns Article,Category '

    if tokenize:
        cmd += '--tokenize '
    if lowercase:
        cmd += '--lowercase '
    if replace_num:
        cmd += '--replace-num '
    if lemmatize:
        cmd += '--lemmatize '

    cmd += f'{input}'

    with open(output, 'w') as f:
        subprocess.check_call(cmd, shell=True, executable='/bin/bash', stdout=f)
