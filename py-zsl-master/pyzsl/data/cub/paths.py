from pathlib import Path

from pyzsl.utils.general import PathBase, as_named_tuple


class CubPaths(PathBase):
    # root
    root         = Path('')

    # config dump
    config = root / 'config.py'

    # directories
    tmp          = Path('tmp')           # files that can be safely removed after generation process exits.
    raw          = Path('raw')           # raw downladed data
    features     = Path('features')      # dense features extracted from the images
    labels       = Path('labels')        # arrays containing label information
    indices      = Path('indices')       # indices mapping a row number in train/dev/test to row number in Descriptions
    descriptions = Path('descriptions')  # final, processed descriptions / attributes
    vocabularies = Path('vocabularies')  # all objects mapping abstract indices to human-readable words
    metadata     = Path('metadata')      # any additional objects describing the data

    # downloaded data, compressed
    orig_c = raw / 'original.tgz'
    reed_c = raw / 'reed.tar.gz'
    xlsa_c = raw / 'xlsa17.zip'

    # raw data, unpacked
    orig_images    = raw / 'CUB_200_2011' / 'images'  # original images of bids
    reed_text      = raw / 'descriptions'       # image descriptions collected by Reed et al
    att_splits_mat = raw / 'att_splits.mat'     # matlab file containing multiple things :(
    trainclasses1  = raw / 'trainclasses1.txt'  # names of classes to use for training
    valclasses1    = raw / 'valclasses1.txt'    # ... for validation
    testclasses    = raw / 'testclasses.txt'    # ... for testing

    # extracted data before splitting

    # images and labels
    resnet_features = as_named_tuple(
        'resnet_features',
        train = features / 'train.npy',
        dev   = features / 'dev.npy',
        test  = features / 'test.npy',
    )

    label_arrays = as_named_tuple(
        'label_arrays',
        train = labels / 'train.npy',
        dev   = labels / 'dev.npy',
        test  = labels / 'test.npy',
    )

    index_arrays = as_named_tuple(
        'index_arrays',
        train = indices / 'train.npy',
        dev   = indices / 'dev.npy',
        test  = indices / 'test.npy',
    )

    # extracted descriptions
    attrs    = descriptions / 'attrs.npy'
    char_lvl = descriptions / 'char_lvl.npy'
    word_lvl = descriptions / 'word_lvl.npy'

    # vocabularies
    word_vocab   = vocabularies / 'word-vocab.dill'
    char_vocab   = vocabularies / 'char-vocab.dill'

    # zsl splits --> assignment of labels to "seen" vs "unseen" groups
    label_stoi        = metadata / 'label-stoi.json'    # mapping label name to label index
    label_itos        = metadata / 'label-itos.json'    # mapping label index to label name
    testset_mask      = metadata / 'testset-masks.npz'  # ['seen', 'unseen']. Binary masks for the testest
    filenames         = metadata / 'filenames.json'


class CubSplits:
    train_ss = 'train_ids'
    valid_ss = 'valid_ids'
    test_ss  = 'test_ids'

    trainval_ps    = 'trainval_ids_ps'
    test_unseen_ps = 'test_unseen_ps'
    test_seen_ps   = 'test_seen_ps'
    test_all_ps    = 'test_all_ps'


class TmpPaths(PathBase):
    root = Path('')
    reed = Path('reed')
    xlsa = Path('xlsa17')

    X = root / 'X.npy'
    Y = root / 'Y.npy'

    text_c10 = reed / 'text_c10'

    att_splits_mat = xlsa / 'data' / 'CUB' / 'att_splits.mat'
    trainclasses1  = xlsa / 'data' / 'CUB' / 'trainclasses1.txt'
    valclasses1    = xlsa / 'data' / 'CUB' / 'valclasses1.txt'
    testclasses    = xlsa / 'data' / 'CUB' / 'testclasses.txt'

    extraction_meta = root / 'meta.pkl'

