import glob
import logging
import os
from collections import Counter, defaultdict
from itertools import chain
from os import path as osp
from pathlib import Path
from typing import Dict, Tuple, Union as U, List, Callable, Union

import numpy as np
import torch as th
from scipy.io import loadmat
from torch import nn, no_grad
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet101, ResNet
from tqdm import tqdm

from pyzsl.data.cub.paths import CubPaths
from pyzsl.utils.general import asdict as _asdict, dinv as _dinv, readlines, \
    dprint
from pyzsl.utils.nlp import nlp_large

PathLike   = Union[Path, str]
Metadata_T = Tuple[Dict[str, int], List[str]]

logger = logging.getLogger(__name__)


class ImageIterator:
    """ yields images and their labels.

        Parameters
        ----------
        root : str
            path
        normalize : bool
            apply image normalization
        size : int, Tuple[int]
            resize

    """

    def __init__(self,
                 root: str,
                 normalize: bool,
                 size: U[int, Tuple[int]]):

        t = [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]

        if normalize:
            t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))

        self.dataset = ImageFolder(root, transform=transforms.Compose(t))
        self.loader = th.utils.data.DataLoader(self.dataset,
                                               batch_size=64, num_workers=7, pin_memory=True, shuffle=False)

    def __iter__(self):
        for x, y in tqdm(self.loader, desc="Image iterator"):
            yield x, y


def load_metadata(root: U[str, Path]) -> Dict[str, np.ndarray]:
    """ Load a matrices from a dataset placed in `root`, which contain several things:
        * attributes of each class
        * indices of items in each split
        * names for each class

    """

    p = CubPaths(root)
    metadata = loadmat(p.att_splits_mat)

    return metadata


def load_attrs(metadata: Dict[str, np.ndarray]) -> np.ndarray:
    """ Given a dict from _load_metadata, extract the attributes. """

    contig = np.ascontiguousarray

    attrs  = metadata['att'].T
    attrs  = contig(attrs).astype(np.float32)

    return attrs


def load_names(metadata: Dict[str, np.ndarray]) -> Dict[str, int]:
    """ Given a dict from _load_metadata, extract the names of each class.

    Return a mapping from class filename to class index.
    """

    names = metadata['allclasses_names']
    names = [name[0][0] for name in names]  # TODO(elan): add note why we [0][0] here

    c2i = _asdict(names)

    return c2i


def load_model_headless(device: str = 'cuda') -> ResNet:
    """ Load a pretrained resnet-101 model from pytorch repo, without the last FC layers """

    class NoOp(nn.Module):
        def forward(self, x):
            return x

    model    = resnet101(pretrained=True)
    model.fc = NoOp()
    model    = model.eval()
    model    = model.to(device)

    return model


def extract_features(root: str,
                     model: ResNet,
                     normalize: bool,
                     size: U[int, Tuple[int, int]],
                     device: str = 'cuda',
                     ) -> Tuple[th.Tensor,
                                th.Tensor,
                                Metadata_T]:

    """ Extraction image features using a deep model.

    Returns
    -------
    features:
        Extracted features
    labels:
        Labels for each row
    meta:
        * a mapping from class filename to int index
        * a list of image filenames that were processed.

    """

    with no_grad():

        model          = model.to(device)
        image_iterator = ImageIterator(root, normalize=normalize, size=size)
        processed      = []

        for x, y in image_iterator:
            x = x.to(device)
            h = model(x).detach().to('cpu')

            processed.append((h, y))

        (features,
         labels) = zip(*processed)

        features = th.cat(features)
        labels   = th.cat(labels)

        meta = (
            _asdict(image_iterator.dataset.class_to_idx),
            [_fname(item[0]) for item in image_iterator.dataset.imgs]  # imgs is a list of tuples (fname, class_idx)
        )

    return features, labels, meta


def remap_labels(labels: th.Tensor,
                 c2i_orig: Dict[str, int],
                 c2i_new: Dict[str, int]) -> th.Tensor:

    """ Given a mapping from label filename to label index returned by running pytorch iterator,
    remap it to the space deifned by 'Good Bad Ugly' code.

    Parameters
    ----------
    labels :
        class index assigned to each image in the dataset
    c2i_orig :
        mapping from class filename to class index as returned by torch
    c2i_new :
        mapping from class filename to class index as provided by Good Bad Ugly code

    Returns
    -------
    ret
        Tensor with re-mapped labels.
    """

    ret      = th.zeros_like(labels)
    i2c_orig = _dinv(c2i_orig)

    for i, y in enumerate(labels):
        y       = y.item()
        name    = i2c_orig[y]
        new_idx = c2i_new[name]

        ret[i] = new_idx

    return ret


def _fname(path: str) -> str:
    """ Return a filename without extension. Make sure file is a jpg or txt. """

    assert path.endswith('.jpg') or path.endswith('.txt')
    return osp.basename(path)[:-4]


def clean_images(data_root):
    data_root = os.path.expanduser(data_root)

    for target in sorted(os.listdir(data_root)):

        d = os.path.join(data_root, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):

                if fname.startswith('._') and fname.endswith('.jpg'):
                    path = os.path.join(root, fname)

                    logger.debug(f'Removing file '
                                 f'of size {os.path.getsize(path)} '
                                 f'at {path}')

                    os.remove(path)


def tokenize_char(desc: str) -> List[str]:
    """ Turn an description into a list of characters """

    return list(desc)


def tokenize_word(desc: str) -> List[str]:
    """ Turn an image description into a list of tokens """

    en = nlp_large()
    return [tok.text for tok in en.tokenizer(desc)]


def process_descriptions(root: str,
                         tokenizer: Callable = tokenize_char) -> Dict[str, List[List[str]]]:
    """ Tokenize descriptions provided by Reed et al. Return a dict mapping
     filename (of file containing the description, without extension), to its tokenized content.

    Parameters
    ----------
    root :
        path to dataset root
    tokenizer :
        Callable to split sentence into tokens.

    Returns
    -------
    Mapping from a filename with a description (without extension) to tokenized content.
    """

    ret = {}
    p    = CubPaths(root, as_str=True)
    desc = p.reed_text

    for path in glob.glob(f'{desc}/*/*.txt'):
        item_name = _fname(path)
        descriptions = [tokenizer(desc.strip())
                        for desc in open(path).readlines()]

        ret[item_name] = descriptions

    return ret


def make_vocab(counter: Counter,
               min_freq: int) -> Dict[str, int]:
    """ Given counts of each token in the dataset,
    replace padding and rare words with special tokens, and return a dictionary
    mapping token string to token index. """

    vocab = defaultdict()
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab.default_factory = lambda vocab=vocab: vocab['<UNK>']

    for k, v in counter.items():
        if v >= min_freq:
            vocab[k] = len(vocab)

    return vocab


def desc_to_tensor(tokenized: Dict[str, List[List[str]]],
                   image_names: List[str],
                   min_freq: int = 1,
                   max_length: int = np.inf) -> Tuple[ th.Tensor,
                                                        Dict[str, int]]:
    """ Convert textual, tokenized descriptions into a torch tensor.

    Parameters
    ----------
    tokenized :
        Mapping from a filename with a description (without extension) to tokenized content.
    image_names :
        a list of image filenames that were processed by the feature extractor. We need to know
        in what order they came.
    min_freq :
        Minimum description lenght. Shorter are discarded
    max_length :
        Maximum description length. Longer are trimmed.

    Returns
    -------
    tensor:
        Descriptions, in the tensor format
    get_vocab:
        A vocabulary mapping token string to its index.
    """

    counter = Counter()

    for fname, descriptions in tokenized.items():
        counter.update(chain(*descriptions))

    vocab      = make_vocab(counter, min_freq=min_freq)
    name_2_idx = _asdict(image_names)

    n_descs  = max(len(descriptions) for descriptions in tokenized.values())
    n_tokens = min(max_length,
                   max(len(d)
                       for descriptions in tokenized.values()
                       for d in descriptions))

    tensor = th.zeros(len(tokenized), n_descs, n_tokens) + vocab['<PAD>']

    # TODO: get rid of this hack...
    def _split(fname):
        """ Class_Name_123_456 -> (Class_Name, 123_456) """
        name, *numbers = fname.rsplit('_', 2)

        name    = name.lower()
        numbers = '_'.join(numbers)

        return name, numbers

    # TODO: get rid of this hack...
    HACKY_mapping = {
        _split(key): key
        for key in name_2_idx
    }

    for fname, descriptions in tokenized.items():

        # TODO: get rid of this hack
        fname = HACKY_mapping[_split(fname)]
        idx   = name_2_idx[fname]

        for i, desc in enumerate(descriptions):
            for j, tok in enumerate(desc):
                tensor[idx, i, j] = vocab[tok]

    return tensor, vocab


def load_xlsa(root: PathLike) -> Tuple[Dict[str, np.ndarray],
                                       np.ndarray,
                                       Dict[str, int]]:

    metadata_xlsa = load_metadata(root)
    attrs_xlsa    = load_attrs(metadata_xlsa)
    c2i_xlsa      = load_names(metadata_xlsa)

    return metadata_xlsa, attrs_xlsa, c2i_xlsa


def run_extraction_pipeline(data_root: PathLike,
                            img_root: PathLike,
                            model: Callable,
                            size: Tuple[int, int],
                            normalize: bool,
                            device: str) -> Tuple[np.ndarray, np.ndarray, Metadata_T]:

    _, _, c2i_xlsa = load_xlsa(data_root)

    (X,
     Y,
     meta) = extract_features(img_root, model, normalize=normalize, size=size, device=device)

    (c2i_orig,
     filenames) = meta

    Y = remap_labels(Y, c2i_orig, c2i_xlsa)
    X = X.numpy()
    Y = Y.numpy()

    return X, Y, meta


def run_parsing_pipeline(filenames: List[str], data_root: PathLike):

    parsed_clvl = process_descriptions(root=data_root, tokenizer=tokenize_char)
    chars, vocab_clvl = desc_to_tensor(parsed_clvl, filenames, min_freq=10)

    parsed_wlvl = process_descriptions(root=data_root, tokenizer=tokenize_word)
    words, vocab_wlvl = desc_to_tensor(parsed_wlvl, filenames, min_freq=3)

    return chars, words, vocab_clvl, vocab_wlvl


def run_splitting_pipeline(X: np.ndarray,
                           Y: np.ndarray,
                           meta: Metadata_T,
                           label_stoi: Dict[str, int],
                           train_classes: PathLike,
                           dev_classes: PathLike,
                           test_classes: PathLike):

    def _load(path):
        return {label_stoi[name] for name in readlines(path)}

    def _arr(ref):
        return np.sort(np.array([i for i in range(Y.shape[0]) if Y[i] in ref]))

    def _ids(splits, key):
        return np.sort(splits[key].reshape(-1) - 1)

    def _get(ids):
        return X[ids, ...], Y[ids, ...]

    train = _load(train_classes)
    dev   = _load(dev_classes)
    test  = _load(test_classes)

    train_ids = _arr(train)
    dev_ids   = _arr(dev)
    test_ids  = _arr(test)

    train_ids_ps    = _ids(meta, 'train_loc')
    dev_ids_ps      = _ids(meta, 'val_loc')
    trainval_ids_ps = _ids(meta, 'trainval_loc')
    test_unseen_ps  = _ids(meta, 'test_unseen_loc')
    test_seen_ps    = _ids(meta, 'test_seen_loc')
    all_test_ids    = np.concatenate([test_seen_ps, test_unseen_ps])

    assert trainval_ids_ps.size == 7057  # from the paper
    assert test_seen_ps.size    == 1764  # from the paper
    assert test_unseen_ps.size  == 2967  # from the paper

    assert np.all(train_ids_ps   == train_ids)
    assert np.all(dev_ids_ps     == dev_ids)
    assert np.all(test_unseen_ps == test_ids)

    X_train, Y_train = _get(train_ids)
    X_dev, Y_dev     = _get(dev_ids)
    X_test, Y_test   = _get(all_test_ids)

    assert X_test.shape[0] == (test_unseen_ps.size + test_seen_ps.size)

    n_seen   = test_seen_ps.size
    n_unseen = test_unseen_ps.size

    seen   = np.zeros((n_seen + n_unseen, ), dtype=np.bool)
    unseen = np.zeros((n_seen + n_unseen, ), dtype=np.bool)

    seen[:n_seen]   = True
    unseen[n_seen:] = True

    Xs    = [X_train, X_dev, X_test]
    Ys    = [Y_train, Y_dev, Y_test]
    IDs   = [train_ids, dev_ids, all_test_ids]
    masks = [seen, unseen]

    assert X_test.shape[0] == seen.shape[0] == unseen.shape[0]

    return Xs, Ys, IDs, masks
