import logging

import torch as th
from attr import s, ib
from ignite.engine import Engine
from torch import optim
from torch.autograd import no_grad
from torch.nn.utils import clip_grad_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from pyzsl.training.cub.pytorch import CubDataset, Descriptions, Splits, CubBatchSampler
from pyzsl.models.deep.cub.sje import SJE, CnnRnn, joint_embedding_loss


@s
class Config:
    data_dir       = ib()
    seed           = ib(123)

    rnn            = ib('rnn')
    init           = ib(None)

    image_dim      = ib(2048)
    emb_dim        = ib(1024)
    cnn_dim        = ib(512)

    img_encoder    = ib(True)
    img_project    = ib(True)
    txt_decoder    = ib(True)
    symmetric      = ib(True)

    dropout        = ib(0.0)
    decay          = ib(0.)
    grad_clip      = ib(0.)

    optim          = ib(optim.RMSprop)
    learning_rate  = ib(0.001)
    lr_decay_step  = ib(1)
    lr_decay_rate  = ib(0.98)

    batch_size     = ib(32)
    doc_length     = ib(256)
    max_epochs     = ib(300)

    cuda           = ib(True)

    print_every     = ib(100)
    validation_freq = ib(25)


def main(cfg: Config):
    logger = logging.getLogger(__name__)
    logger.debug('Entering the execution.')

    th.manual_seed(cfg.seed)

    if cfg.cuda:
        th.cuda.manual_seed(cfg.seed)

    logger.debug('Creating loaders.')

    train_dset    = CubDataset(cfg.data_dir, '_dummy',    max_len=cfg.doc_length)
    valid_dset    = CubDataset(cfg.data_dir, Splits.test_unseen_ps, max_len=cfg.doc_length)
    train_sampler = CubBatchSampler(train_dset, cfg.batch_size)

    train_loader = DataLoader(train_dset,
                              batch_sampler=train_sampler,
                              num_workers=7, pin_memory=True)
    valid_loader = DataLoader(valid_dset,
                              batch_size=cfg.batch_size, shuffle=True, sampler=None,
                              num_workers=7, pin_memory=True)

    logger.debug('Creating cnn-rnn.')

    cnn_rnn = CnnRnn(alphasize=train_loader.dataset.alph_sz,
                     cnn_dim=cfg.cnn_dim,
                     emb_dim=cfg.emb_dim,
                     predictor=cfg.txt_decoder,
                     dropout=cfg.dropout,
                     rnn_type=cfg.rnn)

    logger.debug('Creating SJE.')

    model = SJE(img_size=cfg.image_dim,
                emb_size=cfg.emb_dim,
                img_project=cfg.img_project,
                img_encoder=cfg.img_encoder,
                txt_encoder=cnn_rnn
                )

    if cfg.cuda:
        model = model.cuda()

    if cfg.init is not None:
        for p in model.parameters():
            p.data.uniform_(-cfg.init, cfg.init)

    optimizer = cfg.optim(model.parameters(), weight_decay=cfg.decay, lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_rate)

    _y    = th.arange(start=0, end=cfg.batch_size, step=1, out=th.LongTensor())
    _txt  = Descriptions(valid_dset, cfg.emb_dim, cuda=cfg.cuda)

    if cfg.cuda:
        _y = _y.cuda()

    train_timer = TimeMeter()
    valid_timer = TimeMeter()

    def training_update_function(batch):

        model.train()
        optimizer.zero_grad()

        (img,
         txt,
         _classes) = to_cuda_var(batch, cuda=cfg.cuda, var=True)

        (fea_img,
         fea_txt) = model(img, txt.permute(0, 2, 1))
        y = V(_y)

        loss = joint_embedding_loss(fea_img, fea_txt, y, cfg.symmetric)

        loss.backward()

        if cfg.grad_clip > 0:
            clip_grad_norm(model.parameters(), cfg.grad_clip)

        optimizer.step()

        return float(loss)

    def validation_function(batch):

        with no_grad():
            model.eval()

            (img,
             _,
             _classes) = to_cuda_var(batch, cuda=cfg.cuda, var=True)

            fea_img = model.img_encoder(img)
            scores = fea_img @ _txt.get().t()  # type: th.FloatTensor
            preds  = _txt.to_classes(scores)

            return float((preds == _classes).sum()), _classes.size(0)

    def train_loss_update(engine: Engine):

        train_timer.stop()

        h = trainer.training_history
        ws = len(train_loader)

        value = h.simple_moving_average(window_size=ws)
        scheduler.step()

        ctx.report('best-train', (value, trainer.current_epoch), cmp=min)
        ctx.report('train-time', train_timer.value())
        ctx.log_scalar('train loss', value, trainer.current_epoch)

    def valid_loss_update(engine: Engine):

        h = trainer.validation_history
        ws = len(valid_loader)

        relevant = h[-ws:]
        acc = sum(item[0] for item in relevant)
        cnt = sum(item[1] for item in relevant)

        value = acc / cnt

        ctx.report('best-validation', (value, trainer.current_epoch), cmp=max)
        ctx.log_scalar('test zsl acc', value, trainer.current_epoch)

    def run_validation(engine: Engine):

        if trainer.current_epoch % cfg.validation_freq == 0:
            valid_timer.reset()

            _txt.update(model.txt_encoder)
            trainer.validate(valid_loader)

            valid_timer.stop()
            ctx.report('validation-time', valid_timer.value())

    trainer = Trainer(training_update_function=training_update_function,
                      validation_inference_function=validation_function)

    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_STARTED,   lambda _: train_timer.reset())
    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED, train_loss_update)
    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED, run_validation)
    trainer.add_event_handler(TrainingEvents.VALIDATION_COMPLETED,     valid_loss_update)
    trainer.model = model

    try:
        trainer.run(train_loader, max_epochs=cfg.max_epochs)

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, returning.")

    return trainer
