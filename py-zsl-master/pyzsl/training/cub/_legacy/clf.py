import logging
import os

import torch as th
from attr import s, ib
from ignite.trainer import Trainer, TrainingEvents
from tinydb import TinyDB
from tinydb_smartcache import SmartCacheTable
from torch import optim, nn
from torch.autograd import no_grad
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader

from pyzsl.data.cub.dataset import CubDataset, load
from pyzsl.utils.general import to_cuda_var

TinyDB.table_class = SmartCacheTable


@s
class Config:
    data_dir       = ib()
    storage_dir    = ib()
    seed           = ib(123)

    image_dim      = ib(2048)
    emb_dim        = ib(1024)
    symmetric      = ib(True)
    cnn_dim        = ib(256)
    dropout        = ib(0.0)
    grad_clip      = ib(5)
    learning_rate  = ib(0.001)

    batch_size     = ib(32)
    doc_length     = ib(256)
    max_epochs     = ib(300)

    print_every    = ib(100)
    perm           = ib(False)
    report_train   = ib(True)
    report_valid   = ib(True)
    reed           = ib(False)
    rand           = ib(False)
    log_lvl        = 'info'

    def __attrs_post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)


def main(cfg: Config):
    logger = logging.getLogger(__name__)
    logger.debug('Entering the execution.')

    th.manual_seed(cfg.seed)
    th.cuda.manual_seed(cfg.seed)

    logger.debug('Creating loaders.')

    if cfg.reed:
        cfg.image_dim = 1024

    X, Y = load(root=cfg.data_dir, reed=cfg.reed)

    if cfg.rand:
        mu, sigma = X.mean(), X.std()
        X.normal_(mu, sigma)

    train_loader = DataLoader(CubDataset(X, Y, cfg.data_dir, 'train', 'char_lvl',
                                         max_len=cfg.doc_length,
                                         reed=cfg.reed,
                                         perm=cfg.perm,
                                         rand=cfg.rand),
                              batch_size=cfg.batch_size, shuffle=True, sampler=None,
                              num_workers=7, pin_memory=True)

    valid_loader = DataLoader(CubDataset(X, Y, cfg.data_dir, 'validation', 'char_lvl',
                                         max_len=cfg.doc_length,
                                         reed=cfg.reed),
                              batch_size=cfg.batch_size, shuffle=False, sampler=None,
                              num_workers=7, pin_memory=True)

    logger.debug('Creating clf.')

    model = nn.Sequential(

        nn.Linear(cfg.image_dim, cfg.emb_dim),
        nn.SELU(),
        nn.Dropout(cfg.dropout),

        nn.Linear(cfg.emb_dim, cfg.emb_dim),
        nn.SELU(),
        nn.Dropout(cfg.dropout),

        nn.Linear(cfg.emb_dim, 200)

    ).cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0., lr=cfg.learning_rate)

    def loss_fn(fea_img, _classes):
        loss = F.cross_entropy(fea_img, _classes)
        return loss

    def training_update_function(batch):

        model.train()
        optimizer.zero_grad()

        (img,
         txt,
         _classes) = to_cuda_var(batch, cuda=True, var=True)

        preds = model(img)
        loss  = loss_fn(preds, _classes)

        clip_grad_norm(model.parameters(), cfg.grad_clip)
        loss.backward()
        optimizer.step()

        return float(loss)

    def validation_function(batch):

        with no_grad():

            model.eval()

            (img,
             txt,
             _classes) = to_cuda_var(batch, cuda=True, var=True)

            preds = model(img)

            acc  = float((preds.max(1)[1] == _classes).sum())
            acc /= float(preds.size(0))

            return acc

    def loss_update(trainer: Trainer, db, name, train):

        if train:
            h = trainer.training_history
            ws = len(train_loader)

        else:
            h = trainer.validation_history
            ws = len(valid_loader)

        value = h.simple_moving_average(window_size=ws)

        def _update(doc):
            doc[name].append(value)

        # db.update(_update, where('ID') == ID)

        logger.info(f'Report(name={name}, value={value})')

    logger.debug('Creating Trainer.')

    trainer = Trainer(training_data=train_loader, training_update_function=training_update_function,
                      validation_data=valid_loader, validation_inference_function=validation_function)

    if cfg.report_train:
        trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                                  loss_update, name='train_losses', db=None, train=True)

    if cfg.report_valid:
        trainer.add_event_handler(TrainingEvents.VALIDATION_COMPLETED,
                                  loss_update, name='valid_losses', db=None, train=False)

    logger.info('Invoking Trainer.run()')

    try:
        trainer.run(max_epochs=cfg.max_epochs)

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, returning.")

    return trainer
