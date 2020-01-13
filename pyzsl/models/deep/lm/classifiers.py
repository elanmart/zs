import torch as th

from pyzsl.models.deep.lm.utils import RNNStub
from pyzsl.utils.training import detach


def bpt3c_forward(
        x: th.Tensor,
        rnn: RNNStub,
        bptt: int,
        max_len: int,
        last_only: bool = False
):
    L, B, C = x.size()
    H = rnn.init_hidden()

    outputs = []
    i = 0

    while i < min(L, max_len):
        j    = min(i + bptt, L, max_len)
        seq  = x[i:j, ...]
        O, H = rnn(seq, H)  # O: (j-i, B, C)
        H    = detach(H)

        if last_only:
            O = O[-1, ...].unsqueeze(0)  # O: (1, B, C)

        outputs.append(O)
        i = j

    return th.cat(outputs, dim=0)


def concat_pool(input: th.Tensor,
                lengths: th.Tensor = None):
    """
    input is of shape: (L, B, C)
        L: seq_len, B: batch_sz, C: embed_dim
    lengts: (B, )
        number of elements in each sequence
    """

    L, B, C = input.size()

    if lengths is None:
        hT = input[-1, ...]                   # (B, C)
    else:
        hT = input[lengths, th.arange(B), :]  # (B, C)

    hM = input.max(0)[0]  # (B, C)
    hA = input.mean(0)    # (B, C)

    data = [hT, hM, hA]
    return th.cat(data, dim=1)  # (B, 3 * C)


class PoolingLinearClassifier(th.nn.Module):
    def __init__(self, in_features, layer_sizes, dropout_ps):
        super().__init__()

        layer_sizes   = [3 * in_features] + layer_sizes  # 3 * because of concat pooling
        self.n_layers = len(layer_sizes)

        self.net = th.nn.Sequential(
            th.nn.Sequential(
                LinearBlock(in_features=layer_sizes[i], out_featues=layer_sizes[i + 1], p=dropout_ps[i]),
                th.nn.ReLU(),
            )
            for i in range(self.n_layers)
        )

    def forward(self,
                input: th.Tensor,
                lengths: th.Tensor = None):
        """
        input is of shape: (L, B, C)
            L: seq_len, B: batch_sz, C: embed_dim
        lengts: (B, )
            number of elements in each sequence
        """

        x = concat_pool(input=input, lengths=lengths)
        x = self.net(x)

        return x


class LinearBlock(th.nn.Module):
    def __init__(self, in_features, out_featues, p):
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.BatchNorm1d(in_features),
            th.nn.Dropout(p),
            th.nn.Linear(in_features, out_featues),
        )

    def forward(self, input):
        return self.net(input)
