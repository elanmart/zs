from typing import Tuple, Union, List

import numpy as np
import torch as th


class LMLoader:
    def __init__(
        self,
        source: Union[th.Tensor, np.ndarray],
        device: str = "cpu",
        bptt: int = 10,
        batch_size: int = 20,
        evaluation: bool = False,
        to_device: bool = False,
    ):
        self.evaluation = evaluation
        self.bptt = bptt
        self.batch_size = batch_size
        self.device = device
        self.to_device = to_device

        if isinstance(source, np.ndarray):
            source = th.from_numpy(source)

        data = source.data.long()
        self.batches = self.batchify(data, batch_size)

    def batchify(self, data: th.Tensor, bsz: int) -> th.Tensor:
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()

        if self.to_device:
            data = data.to(self.device)

        return data

    def get_batch(self, i: int) -> Tuple[th.Tensor, th.Tensor]:
        seq_len = min(self.bptt, len(self.batches) - 1 - i)
        data    = self.batches[i: i + seq_len]
        target  = self.batches[i + 1: i + 1 + seq_len].view(-1)

        return (
            data.to(self.device, non_blocking=True),
            target.to(self.device, non_blocking=True)
        )

    def __len__(self):
        return self.batches.size(0) // self.bptt

    def __iter__(self):
        for i in range(0, self.batches.size(0) - 1, self.bptt):
            yield self.get_batch(i)
