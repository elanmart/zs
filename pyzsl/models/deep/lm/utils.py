import logging
from typing import Tuple, List, Type, Optional

import numpy as np
import torch as th
from torch.nn import AdaptiveLogSoftmaxWithLoss as _AdaptiveLogSoftmaxWithLoss
from torch.nn import Linear, ModuleList

logger = logging.getLogger(__name__)
FREQ = 250


class RNNStub(th.nn.Module):
    def forward(self,
                input: th.Tensor,
                hidden: Optional[Tuple[th.Tensor]]= None):
        raise NotADirectoryError()

    def init_hidden(self) -> Optional[Tuple[th.Tensor]]:
        raise NotADirectoryError()

    def reset(self) -> None:
        raise NotADirectoryError()

    def get_layer_params(self) -> List[List[th.nn.Parameter]]:
        raise NotADirectoryError()


class AdaptiveLogSoftmaxWithLoss(_AdaptiveLogSoftmaxWithLoss):
    def tie_weights(self, W):
        self.head.weight[:self.shortlist_size] = W[:self.shortlist_size]
        self.tail = ModuleList()

        for i in range(self.n_clusters):
            i0  = self.cutoffs[i + 1]
            i1  = self.cutoffs[i]
            osz = i0 - i1

            projection = Linear(self.in_features, osz, bias=False)
            projection.weight = W[i0:i1]

            self.tail.append(projection)


class STLRScheduler(th.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: th.optim.Optimizer,
            T: int,
            cut_frac: float = 0.1,
            ratio: int = 32,
            last_epoch: int = -1,
    ):
        """ Implements Slanted triangular learning rates from https://arxiv.org/pdf/1801.06146.pdf
        LR_max from the paper is the learning_rate supplied to the optimizer!!!

        Parameters
        ----------
        optimizer :
            optimizer to schedule
        T :
            the number of epochs times the number of updates per epoch
        cut_frac :
             fraction of iterations we increase the LR
        ratio :
            specifies how much smaller the lowest LR is from the LR_max
        last_epoch :
            see docs of base class
        """

        self.T        = T
        self.cut_frac = cut_frac
        self.ratio    = ratio

        self._cut = int(T * cut_frac)

        super().__init__(optimizer, last_epoch)

    def _get_single_lr(self, LR_max):
        t = self.last_epoch
        c = self._cut
        LR_min = LR_max / self.ratio

        if t < c:
            slope = (LR_max - LR_min) / c
            lr    = slope * t + LR_min

        else:
            t     = t - c
            slope = -(LR_max - LR_min) / (self.T - c)
            lr    = slope * t + LR_max

        return lr

    def get_lr(self):
        return [
            self._get_single_lr(lr_max)
            for lr_max in self.base_lrs
        ]


def get_discriminative_optimizer(cls: Type[th.optim.Adam],  # use Adam here instead of Optimizer (O. doesnt take lr arg)
                                 layered_params: List[List[th.nn.Parameter]],
                                 base_lr: float,
                                 discount_factor: float,
                                 *args, **kwargs):
    params = [
        {'params': p, 'lr': base_lr / (discount_factor ** i)}
        for i, p in enumerate(layered_params)
    ]

    optimizer = cls(params=params, lr=base_lr, *args, **kwargs)
    return optimizer


def get_weight_matrix(W: th.Tensor, old_itos: List[str], new_itos: List[str]):
    n_tokens = len(new_itos)
    emb_size = W.size(1)

    avg = W.mean(0)
    ret = W.new_zeros(n_tokens, emb_size) + avg.reshape(1, -1)

    new_stoi  = {s: i for i, s in enumerate(new_itos)}
    old_2_new = {  # maps an index of original vocab to index in new vocab, if a token appears in both
        old_ix: new_stoi[old_str]
        for old_ix, old_str in enumerate(old_itos)
        if old_str in new_stoi
    }

    for old, new in old_2_new.items():
        ret[new] = W[old]

    return ret


def remap(arr, mapping=None):
    uq, cnt = np.unique(arr, return_counts=True)
    inds    = np.argsort(-cnt)

    if mapping is None:
        mapping = {}

        for i in inds:
            idx = uq[i]
            mapping[idx] = len(mapping)

    ret = [mapping[item] for item in arr]
    ret = np.array(ret, dtype=arr.dtype)

    return ret, mapping


def set_weight(module, W, check_size=False, clone=False):
    if clone:
        W = W.clone().detach()

    if not isinstance(W, th.nn.Parameter):
        W = th.nn.Parameter(W)

    if isinstance(module, th.nn.Linear):
        assert module.bias is None, "cannot set weight for linear with bias"

        if check_size:
            assert module.in_features == W.size(1)
            assert module.out_features == W.size(0)

        module.out_features = W.size(0)
        module.in_features = W.size(1)
        module.weight = W

    elif isinstance(module, th.nn.Embedding):
        if check_size:
            assert module.num_embeddings == W.size(0)
            assert module.embedding_dim == W.size(1)

        module.num_embeddings = W.size(0)
        module.embedding_dim = W.size(1)
        module.weight = W

    elif isinstance(module, th.nn.AdaptiveLogSoftmaxWithLoss):

        assert module.head_bias is False

        in_features = W.size(1)
        n_classes = W.size(0)
        cutoffs = module.cutoffs[:-1]

        module.in_features = in_features
        module.n_classes = n_classes
        module.cutoffs = cutoffs + [n_classes]

        module.shortlist_size = module.cutoffs[0]
        module.head_size = module.shortlist_size + module.n_clusters

        module.head.weight[:module.shortlist_size] = W[:module.shortlist_size]
        module.tail = ModuleList()

        for i in range(module.n_clusters):
            i0 = module.cutoffs[i + 1]
            i1 = module.cutoffs[i]
            osz = i0 - i1

            projection = Linear(module.in_features, osz, bias=False)
            projection.weight = W[i0:i1]

            module.tail.append(projection)

    elif hasattr(module, 'weight'):
        module.weight = W

    else:
        raise RuntimeError(f"idk how to set the weight for module "
                           f"{module}: it is not a [Linear, Embedding, Adaptive]")
