import torch
import torch as th
from torch import autograd
from torch import nn
from torch.autograd import Variable as V


class Generator(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=1024, hid_dim=4096, z_dim=2048):
        super().__init__()

        in_dim      = z_dim + txt_dim
        self._z_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, img_dim),
            nn.ReLU()
        )

    def sample(self, txt):

        noise = th.randn(txt.size(0), self._z_dim).type_as(txt)
        input = th.cat([noise, txt], dim=1).detach()

        fake_img = self(input)

        return fake_img

    def forward(self, x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=1024, hid_dim=4096):
        super().__init__()

        in_dim = img_dim + txt_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_in, n_out, n_hid=None, mlp=False):
        super().__init__()

        if mlp:
            self.model = nn.Sequential(
                nn.Linear(n_in, n_hid),
                nn.LeakyReLU(),
                nn.Linear(n_hid, n_out)
            )

        else:
            self.model = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = self.model(x)

        return x


def _grad_penalty(discriminator, real_data, fake_data, lbd):

    real_data = real_data.data
    fake_data = fake_data.data

    alpha = real_data.new(real_data.size(0), 1).uniform_(0, 1)
    alpha = alpha.expand(real_data.size(0), real_data.size(1)).contiguous()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates      = V(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)
    grad_outputs      = torch.ones_like(disc_interpolates.data)

    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    ret = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lbd  # type: torch.FloatTensor

    return ret
