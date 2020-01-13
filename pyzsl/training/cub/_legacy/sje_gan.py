from itertools import chain
from os.path import join
from pathlib import Path

import gc

import bnb
import torch as th
from attr import ib, attrs
from bnb import Experiment
from ignite.engine import Events
from ignite.evaluator import Evaluator
from ignite.trainer import Trainer
from pyzsl.models.deep.cub.sje import joint_embedding_loss
from torch import no_grad
from torch.utils.data import DataLoader

from pyzsl.data.cub import CubSplits
from pyzsl.data.cub.dataset import (CubDataset, Descriptions, GanDataset)
from pyzsl.models.deep.cub import (Generator, Discriminator, _grad_penalty, Classifier)
from pyzsl.utils.general import (to_cuda_var, TimeMeter)
from pyzsl.training.cub.sje_rev import main as rev_main, experiment as rev_ex, Config as rev_Config

experiment = Experiment('pyzsl-gan', 'gan', dirty_ok=True)

best_cfg = rev_Config(**{'average': True,  'batch_size': 40, 'cnn_dim': 512, 'cuda': True,
                         'data_dir': Path('/media/elan/SAMSUNG/storage/py-zsl/cub-final'),
                         'decay': 0.0,  'dist_inner_dim': 512, 'dist_nlayers': 2.0, 'doc_length': 201, 'dropout': 0.0,
                         'emb_dim': 2048, 'grad_clip': 0.0, 'image_dim': 2048, 'img_bias': True, 'img_nlayers': 1.0,
                         'init_range': 0.05, 'learning_rate': 0.001,  'loss': 'sje', 'lr_decay_rate': 0.98,
                         'lr_decay_step': 1.0, 'max_epochs': 151, 'normalize_img': False, 'normalize_txt': False,
                         'optim': th.optim.Adam, 'repr_loss': 'sje', 'rev_alpha': 0.5, 'rnn': 'lstm', 'seed': 123,
                         'swap_images': False, 'symmetric': True, 'txt_decoder': False, 'use_zscore': False,
                         'validation_freq': 150})


@attrs
class Config:
    model_path = Path('/tmp/cnn-rnn-model.pth')

    batch_size = ib(64)
    mlp        = ib(False)

    lr         = ib(0.001)

    beta       = ib(0.01)
    lbd        = ib(10)
    D_steps    = ib(5)

    gp         = ib(True)

    max_epochs = ib(300)

    save_interval = ib(25)
    eval_interval = ib(25)

    z_dim   = ib(best_cfg.emb_dim)
    use_txt = ib(True)
    use_z   = ib(True)


def get_model(path: Path, cfg=best_cfg, force=False):

    if path.exists() and force is False:
        model = th.load(path)

    else:
        with rev_ex.call():
            trainer = rev_main(cfg)

        model = trainer.model

        th.save(model, path)

    return model


def _set_grad(net: th.nn.Module, flag: bool):
    for p in net.parameters():
        p.requires_grad = flag

    return net


@experiment.watch
def train_gan(cfg: Config):

    gc.collect()
    th.cuda.empty_cache()

    # PREP -------------------------------------------------------------------------------------------------------------

    ctx = bnb.get_current_context()

    th.manual_seed(best_cfg.seed)
    th.cuda.manual_seed(best_cfg.seed)

    # DATA -------------------------------------------------------------------------------------------------------------
    model = get_model(cfg.model_path, cfg=best_cfg).cuda()

    _train_dset = CubDataset(best_cfg.data_dir,
                             ids_subset=CubSplits.trainval_ps,
                             max_len=best_cfg.doc_length,
                             swap_image=best_cfg.swap_images)

    _valid_dset = CubDataset(best_cfg.data_dir,
                             ids_subset=CubSplits.trainval_ps,
                             max_len=best_cfg.doc_length)

    train_desc = Descriptions(_train_dset, best_cfg.emb_dim, loss='sje', norm=None, no_norm=True).cuda()
    train_desc.update(model.txt_encoder)
    train_desc = train_desc.cpu()

    valid_desc = Descriptions(_valid_dset, best_cfg.emb_dim, loss='sje', norm=None, no_norm=True).cuda()
    valid_desc.update(model.txt_encoder)
    valid_desc = valid_desc.cpu()

    model = model.cpu()
    enc = model.img_encoder.cuda()

    train_dset = GanDataset(train_desc)
    valid_dset = GanDataset(valid_desc)

    train_loader = DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid_dset, batch_size=cfg.batch_size, shuffle=True, num_workers=3, pin_memory=True)

    # MODELS -----------------------------------------------------------------------------------------------------------

    generator = Generator(img_dim=best_cfg.emb_dim,
                          txt_dim=best_cfg.emb_dim,
                          hid_dim=best_cfg.emb_dim,
                          z_dim=cfg.z_dim,
                          txt=cfg.use_txt,
                          z=cfg.use_z).cuda()

    discriminator = Discriminator(img_dim=best_cfg.emb_dim,
                                  txt_dim=best_cfg.emb_dim,
                                  hid_dim=best_cfg.emb_dim,
                                  txt=cfg.use_txt).cuda()

    clf = Classifier(n_in=best_cfg.emb_dim,
                     n_out=_train_dset.n_classes,
                     n_hid=best_cfg.emb_dim + cfg.z_dim,
                     mlp=cfg.mlp).cuda()

    optimizer_G = th.optim.Adam(chain(generator.parameters(), clf.parameters()),
                                betas=(0.5, 0.99),
                                lr=cfg.lr)

    optimizer_D = th.optim.Adam(discriminator.parameters(),
                                betas=(0.5, 0.99),
                                lr=cfg.lr)

    # MISC -------------------------------------------------------------------------------------------------------------

    train_timer = TimeMeter()
    valid_timer = TimeMeter()

    # HOOKS ------------------------------------------------------------------------------------------------------------

    def training_update_function(batch):

        # ----------------------
        # Setup

        def _D_input(fea_img, fea_txt):
            if cfg.use_txt:
                return th.cat([fea_img, fea_txt], dim=1)
            return fea_img

        generator.cuda().train()
        discriminator.cuda().train()

        # model.cpu()
        valid_desc.cpu()

        (fea_img,
         fea_txt,
         y) = to_cuda_var(batch, cuda=best_cfg.cuda, var=True)

        # ----------------------
        # Update discriminator

        _set_grad(discriminator, True)
        discriminator.zero_grad()
        generator.zero_grad()

        with th.no_grad():
            fake_img = generator.sample(fea_txt)
            real = _D_input(fea_img,  fea_txt)
            fake = _D_input(fake_img, fea_txt)

        D_real = discriminator(real).mean()
        D_fake = discriminator(fake).mean()

        gp = 0.
        if cfg.gp:
            gp = _grad_penalty(discriminator, real, fake, cfg.lbd)

        D_loss = -(D_real - D_fake - gp)
        Wasserstein_D = D_real - D_fake

        D_loss.backward()
        optimizer_D.step()

        D_real = joint_embedding_loss(enc(fea_img), fea_txt)
        D_fake = joint_embedding_loss(enc(fake_img), fea_txt)
        D_loss = D_real - D_fake

        # ----------------------
        # Update generator only once every D_steps

        if ((trainer.current_iteration + 1) % cfg.D_steps) != 0:
            return [float(x) for x in
                    (D_fake, D_real, D_loss, Wasserstein_D, 0., gp)]

        # ----------------------
        # Update generator

        _set_grad(discriminator, False)
        generator.zero_grad()

        fake_img = generator.sample(fea_txt)
        fake     = _D_input(fake_img, fea_txt)

        G = -discriminator(fake).mean()
        G.backward()
        optimizer_G.step()

        return [float(x) for x in
                (D_fake, D_real, D_loss, Wasserstein_D, G, gp)]

    def validation_function(batch):

        generator.cuda()
        discriminator.cpu()

        model.img_encoder.cuda()
        model.eval()
        valid_desc.cuda()

        with no_grad():
            (real_img,
             fea_txt,
             y) = batch

            (real_img,
             fea_txt,
             y)      = to_cuda_var((real_img, fea_txt, y), cuda=True, var=True)
            fake_img = generator.sample(fea_txt)

            def _stats(img):
                img  = model.img_encoder(img)
                p, s = valid_desc.predict2(img)
                ent  = -th.sum(th.log(s) * s)
                hits = float((p == y).sum())

                return hits, ent

            real_hits, real_ent = _stats(real_img)
            fake_hits, fake_ent = _stats(fake_img)

            return real_hits, fake_hits, real_ent / fake_ent, fea_txt.size(0)

    def train_loss_update(trainer: Trainer):
        train_timer.stop()

        h = trainer.history
        ws = len(train_loader)

        D_fake = h.simple_moving_average(window_size=ws, transform=lambda x: x[0])
        D_real = h.simple_moving_average(window_size=ws, transform=lambda x: x[1])
        D_loss = h.simple_moving_average(window_size=ws, transform=lambda x: x[2])
        Wasserstein_D = h.simple_moving_average(window_size=ws, transform=lambda x: x[3])
        G_loss = h.simple_moving_average(window_size=ws, transform=lambda x: x[4]) * cfg.D_steps
        gp = h.simple_moving_average(window_size=ws, transform=lambda x: x[5])

        ctx.report('train-time', train_timer.value())

        ctx.log_scalar('D_fake', D_fake, trainer.current_epoch)
        ctx.log_scalar('D_real', D_real, trainer.current_epoch)
        ctx.log_scalar('D_loss', D_loss, trainer.current_epoch)
        ctx.log_scalar('Wasserstein_D', Wasserstein_D, trainer.current_epoch)
        ctx.log_scalar('G_loss', G_loss, trainer.current_epoch)
        ctx.log_scalar('gp', gp, trainer.current_epoch)

    def valid_loss_update(ev: Evaluator, tr: Trainer):
        acc_r = sum(item[0] for item in ev.history)
        acc_f = sum(item[1] for item in ev.history)
        ent   = sum(item[2] for item in ev.history)
        cnt   = sum(item[3] for item in ev.history)

        acc_r = acc_r / float(cnt)
        acc_f = acc_f / float(cnt)
        ent   = ent / len(ev.history)

        ctx.report('validation-time', valid_timer.value())

        ctx.log_scalar('test-gan-acc-r', acc_r, tr.current_epoch)
        ctx.log_scalar('test-gan-acc-f', acc_f, tr.current_epoch)
        ctx.log_scalar('test-gan-ent',   ent,   tr.current_epoch)

        ev.history.clear()

    def run_validation(trainer, evaluator):
        if (trainer.current_epoch % cfg.eval_interval) == 0:
            valid_timer.reset()
            evaluator.run(valid_loader)

    def save(trainer, path):
        if (trainer.current_epoch % cfg.save_interval) == 0:
            th.save(train_desc.cpu(),            join(path, 'descriptions.pth'))
            th.save(generator.cpu().eval(),      join(path, 'generator.pth'))
            th.save(discriminator.cpu().eval(),  join(path, 'discriminator.pth'))

    # SETUP
    trainer = Trainer(training_update_function)
    evaluator = Evaluator(validation_function)

    # TRAINING HANDLERS
    trainer.add_event_handler(Events.EPOCH_STARTED,    lambda _: train_timer.reset())
    trainer.add_event_handler(Events.EPOCH_COMPLETED,  train_loss_update)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,  save, '/tmp')

    # VALIDATION HANDLERS
    trainer.add_event_handler(Events.EPOCH_COMPLETED,  run_validation, evaluator)
    evaluator.add_event_handler(Events.COMPLETED,      valid_loss_update, trainer)

    # RUN
    trainer.run(train_loader, max_epochs=cfg.max_epochs)
