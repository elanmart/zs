""" Modified from https://github.com/locuslab/TCN """


import torch.nn as nn
from torch.nn.utils import weight_norm

from pyzsl.models.deep.lm.dropout import embedded_dropout


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class NoOp(nn.Module):
    def forward(self, input):
        return input


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )

        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2  = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )

        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.downsample = NoOp()

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if hasattr(self.downsample, 'weight'):
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels   = num_inputs if i == 0 else num_channels[i-1]
            out_channels  = num_channels[i]

            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 lock_emb=True,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super().__init__()

        self.emb_dropout = emb_dropout
        self.lock_emb    = lock_emb

        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn     = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.drop_e  = nn.Dropout(emb_dropout)

        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)

    def init_hidden(self, *args, **kwargs):
        return None

    def reset(self):
        return

    def get_layer_params(self):
        return [
            list(self.encoder.parameters()),
            *(
                list(module.parameters())
                for module in self.tcn.network
            )
        ]

    def forward(self, input, hidden=None):
        """ Input ought to have dimension (N, C, L),
            where L is the seq_len; here the input is (L, N)

            hidden is dummy arg so that this becomes a drop-in replacement for RNNs.
        """

        assert hidden is None, "what?"

        if self.lock_emb:
            p   = self.emb_dropout * self.training
            emb = embedded_dropout(self.encoder, input, dropout=p)

        else:
            emb = self.encoder(input)
            emb = self.drop_e(emb)

        # emb: (L, N, C)
        emb = emb.permute(1, 2, 0)  # (N, C, L)
        y   = self.tcn(emb)         # (
        y   = y.transpose(1, 2)     # (

        return y.contiguous(), hidden
