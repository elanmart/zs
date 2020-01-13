import logging
from typing import Tuple, List, Union, Iterator

import torch as th
from ignite.metrics import Metric

logger = logging.getLogger(__name__)


def detach(H: Union[List, Tuple, th.Tensor]):
    if H is None:
        return

    typ = type(H)

    if typ in {list, tuple}:
        return typ(detach(h) for h in H)

    return H.detach()


def ckpt_noop(fn, *args):
    return fn(*args)


def freeze_(layer):
    layer.train(False)

    for p in layer.parameters():
        p.requires_grad = (False)
        p.grad = None

    return layer


def unfreeze_(layer):
    layer.train(True)
    for p in layer.parameters():
        p.requires_grad = (True)

    return layer


class Average(Metric):

    def __init__(self):
        super().__init__()
        self._stat = 0.
        self._cnt  = 0

    def reset(self):
        self._stat = 0.
        self._cnt  = 0

    def update(self, output):
        output = float(output)
        self._stat += output
        self._cnt  += 1

    def update_many(self, sum, cnt):
        self._stat += sum
        self._cnt += cnt    

    def compute(self):
        return self._stat / self._cnt


class MovingAverage(Metric):

    def __init__(self, p):
        super().__init__()

        self._p     = p
        self._stat = 0.

    def reset(self):
        self._stat = 0.

    def update(self, output):
        output = float(output)
        self._stat = self._p * self._stat + (1 - self._p) * output

    def compute(self):
        return self._stat


class Unfreezer:
    def __init__(self,
                 layers: List[th.nn.Module],
                 reverse: bool = False):

        self.layers   = reversed(layers) if reverse else layers
        self.iterator = iter(layers)  # type: Iterator[th.nn.Module]

        for layer in self.layers:
            freeze_(layer)

    def step(self):
        try:
            layer = next(self.iterator)

        except StopIteration:
            pass

        else:
            logger.info('Unfreezing...')
            unfreeze_(layer)
