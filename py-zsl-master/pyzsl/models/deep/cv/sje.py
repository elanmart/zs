import torch as th
from torch import nn
from torch.nn import functional as F


class GradReversal(th.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha=1.):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.alpha * grad_output), None


class NoOp(nn.Module):
    def forward(self, input):
        return input


class ImageEncoder(nn.Module):
    def __init__(self, *, img_dim=None, emb_dim=None,
                 bias=True, n_layers=0, dropout=0.):

        super().__init__()

        if n_layers == 0:
            self.extractor = NoOp()
            self.predictor = NoOp()

        elif n_layers == 1:
            self.extractor = nn.Linear(img_dim, emb_dim, bias=bias)
            self.predictor = NoOp()

        else:
            self.extractor = nn.Sequential(
                nn.Linear(img_dim, emb_dim, bias=bias),
                nn.SELU(),
                nn.Dropout(dropout),
            )
            self.predictor = nn.Linear(emb_dim, emb_dim, bias=bias)

    def forward(self, image):
        x = self.extractor(image)
        x = self.predictor(x)

        return x


class CnnRnn(nn.Module):

    def __init__(self, alphasize, cnn_dim, emb_dim,
                 dropout=0., rnn_type='lstm', predictor=True, average=True):

        super().__init__()

        cnn = nn.Sequential(
            nn.Conv1d(alphasize, 384, 4),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),

            nn.Conv1d(384, 512, 4),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),

            nn.Conv1d(512, cnn_dim, 4),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
        )

        assert rnn_type in {'lstm', 'rnn'}

        if rnn_type == 'lstm':
            rnn = nn.LSTM(input_size=cnn_dim, hidden_size=cnn_dim,
                          batch_first=True)
        else:
            rnn = nn.RNN(input_size=cnn_dim, nonlinearity='relu',
                         hidden_size=cnn_dim, batch_first=True)

        if predictor:
            predictor = nn.Sequential(
                nn.Dropout(dropout),

                nn.Linear(cnn_dim, emb_dim),
                nn.SELU(),
                nn.Dropout(dropout),

                nn.Linear(emb_dim, emb_dim))

        else:
            predictor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(cnn_dim, emb_dim)
            )

        self.rnn_type = rnn_type
        self.avg = average

        self.cnn = cnn
        self.rnn = rnn

        self.predictor = predictor

    def extractor(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        (o,
         h) = self.rnn(x)

        if self.rnn_type == 'lstm':
            (h, _) = h

        if self.avg:
            x = o.mean(1)
        else:
            x = h.squeeze()

        return x

    def forward(self, x):
        x = self.extractor(x)
        x = self.predictor(x)

        return x


class SJE(nn.Module):
    def __init__(self, img_encoder, txt_encoder,
                 normalize_img=False, normalize_txt=False):

        super().__init__()

        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder

        self.normalize_img = normalize_img
        self.normalize_txt = normalize_txt

    def forward(self, images, descriptions):
        fea_img = self.img_encoder(images)
        fea_txt = self.txt_encoder(descriptions)

        if self.normalize_img:
            fea_img = F.normalize(fea_img, p=2, dim=1)

        if self.normalize_txt:
            fea_txt = F.normalize(fea_txt, p=2, dim=1)

        return fea_img, fea_txt


class Distinguisher(nn.Module):
    def __init__(self, emb_dim, inner_dim=None, nlayers=1, dropout=0.):
        super().__init__()

        if nlayers == 1:
            self.model = nn.Linear(emb_dim, 1)

        elif nlayers == 2:
            self.model = nn.Sequential(
                nn.Linear(emb_dim, inner_dim),
                nn.SELU(),
                nn.Dropout(dropout),

                nn.Linear(inner_dim, 1)
            )

        else:
            self.model = nn.Sequential(
                nn.Linear(emb_dim, inner_dim),
                nn.SELU(),
                nn.Dropout(dropout),

                nn.Linear(inner_dim, inner_dim),
                nn.SELU(),
                nn.Dropout(dropout),

                nn.Linear(inner_dim, 1)
            )

    def forward(self, x):
        return self.model(x)


def reverse_grad(x, alpha):
    return GradReversal.apply(x, alpha)


def _joint_embedding_loss(fea_img, fea_txt):
    batch_shape = fea_img.shape[0]
    num_class   = fea_txt.shape[0]

    score  = fea_img @ fea_txt.t()
    score -= th.diag(score).view(-1, 1)

    thresh  = score + 1
    thresh -= th.eye(batch_shape, out=score.new(batch_shape, batch_shape))
    thresh.clamp_(min=0)

    loss = thresh.sum()
    denom = (batch_shape * num_class)  # - min(batch_shape, num_class)  So that we're comparable with Scott's code

    return loss / denom


def joint_embedding_loss(fea_img, fea_txt, symmetric):
    loss = _joint_embedding_loss(fea_img, fea_txt)

    if symmetric:
        loss += _joint_embedding_loss(fea_txt, fea_img)

    return loss


def cosine_loss_fn(fea_img, fea_txt):
    raise NotImplementedError()


def triplet_loss_fn():
    raise NotImplementedError()
