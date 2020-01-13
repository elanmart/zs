from pathlib import Path
from pyzsl.utils.general import PathBase, as_named_tuple

class AWAPaths(PathBase):
    """ Warning: many original files use 1 based indexing """
    root = Path('')

    # directories
    tmp          = Path('tmp')
    raw          = Path('raw')
    features     = Path('features')
    labels       = Path('labels')
    indices      = Path('indices')
    descriptions = Path('descriptions')
    metadata     = Path('metadata')

    base_zip     = raw / "AwA2-base.zip"
    features_zip = raw / "AwA2-features.zip"
    xlsa_zip     = raw / "xlsa17.zip"

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

    # Probably it won't be very useful
    index_arrays = as_named_tuple(
        'index_arrays',
        train = indices / 'train.npy',
        dev   = indices / 'dev.npy',
        test  = indices / 'test.npy',
    )

    # Per class attributes (binarized or couninuous)
    attrs_bin  = descriptions / 'attrs-bin.npy'
    attrs_cont = descriptions / 'attrs-cont.npy'
    attrs = attrs_bin                                   # use bin matrix by default

    label_stoi        = metadata / 'label-stoi.json'    # mapping label name to label index
    label_itos        = metadata / 'label-itos.json'    # mapping label index to label name
    attrs_stoi        = metadata / 'attrs-stoi.json'
    attrs_itos        = metadata / 'attrs-itos.json'

    testset_mask      = metadata / 'testset-masks.npz'  # ['seen', 'unseen']. Binary masks for the testest



class TmpPaths(PathBase):
    root = Path('.')

    # unzipped dirs
    base_dir = root / 'Animals_with_Attributes2'
    features_dir = base_dir / 'Features' / 'ResNet101'
    xlsa_dir = root / 'xlsa17'

    # Base stuff
    classes_txt       = base_dir / 'classes.txt'
    attrs_txt         = base_dir / 'predicates.txt'
    attrs_matrix_bin  = base_dir / 'predicate-matrix-binary.txt'
    attrs_matrix_cont = base_dir / 'predicate-matrix-continuous.txt'

    # Features + labels
    features = features_dir / 'AwA2-features.txt'
    labels   = features_dir / 'AwA2-labels.txt'

    # Split from XLSA
    trainval_classes = xlsa_dir / 'data' / 'AWA2' / 'trainvalclasses.txt'
    test_classes     = xlsa_dir / 'data' / 'AWA2' / 'testclasses.txt'
