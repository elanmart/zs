import logging

import torch as th
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def similarity(x: th.Tensor, y: th.Tensor, mode: str):
    mapping = {
        'euclidean': euclidean_similarity,
        'sje':       sje_similarity,
        'cosine':    cosine_similarity,
    }

    assert mode in mapping, "unknown mode"

    fn = mapping[mode]
    return fn(x, y)


def euclidean_similarity(x: th.Tensor, y: th.Tensor):
    XX = (x.norm(p=2., dim=1) ** 2).unsqueeze(1)
    YY = (y.norm(p=2., dim=1) ** 2).unsqueeze(0)

    distances = x @ y.t()
    distances *= -2
    distances += XX
    distances += YY
    distances *= -1  # because we want to maximize cosine and sje

    return distances


def sje_similarity(x: th.Tensor, y: th.Tensor):
    return x @ y.t()


def cosine_similarity(x: th.Tensor, y: th.Tensor):
    return F.normalize(x) @ F.normalize(y).t()


def adjust_scores(scores, mu, sigma) -> th.Tensor:
    scores  = (scores - mu) / sigma
    _, inds = scores.max(1)

    return Variable(self.Y)[inds]
