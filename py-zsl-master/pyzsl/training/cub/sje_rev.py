import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict

import fire
import numpy
import torch as th
from attr import ib, attrs, validators
from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
from torch import optim
from torch.autograd import no_grad
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from pyzsl.data.cub.paths import CubPaths
from pyzsl.models.deep.cv.similarities import similarity
from pyzsl.models.deep.cv.sje import ImageEncoder, CnnRnn, SJE, reverse_grad
from pyzsl.models.deep.cv.sje import joint_embedding_loss, Distinguisher
from pyzsl.training.cub.loaders import CubDataset, Descriptions
from pyzsl.training.cub.loaders import CubBatchSampler, TransferDataset, TransferLoader
from pyzsl.utils.general import ChainedParams
from pyzsl.utils.training import MovingAverage, Average


def _filter_values(dct: Dict):
    dct = {k: float(v) for k, v in dct.items()}

    return dct


def _write_row(row, f):
    row = ';'.join(str(item) for item in row)
    print(row, file=f, flush=True)


def _set_log_path(p):
    p = Path(p)
    assert not p.is_dir()
    p.parent.mkdir(parents=True, exist_ok=True)

    return p


def _compute(value):
    if hasattr(value, 'compute'):
        return value.compute()
    if int(value) == float(value):
        return int(value)
    return float(value)


@attrs
class Config:
    data_dir       = ib()  # type: CubPaths
    log_path       = ib()  # type: str
    dev_log_path   = ib()  # type: str

    seed           = ib(123)

    symmetric      = ib(True)

    rnn            = ib('rnn')
    init_range     = ib(None)

    emb_dim        = ib(2048)

    img_bias       = ib(True)
    img_nlayers    = ib(0)
    image_dim      = ib(2048)

    txt_decoder    = ib(True)
    cnn_dim        = ib(512)
    average        = ib(True)

    dropout        = ib(0.0)
    decay          = ib(0.)
    grad_clip      = ib(0.)

    optim          = ib(optim.RMSprop)
    learning_rate  = ib(0.001)
    lr_decay_step  = ib(1)
    lr_decay_rate  = ib(0.98)

    batch_size     = ib(32)
    doc_length     = ib(201)
    max_epochs     = ib(300)

    rev_alpha      = ib(0.5)
    dist_nlayers   = ib(2)
    dist_inner_dim = ib(512)

    normalize_repr  = ib(True)
    normalize_img   = ib(False)
    normalize_txt   = ib(False)

    use_zscore      = ib(False)

    loss            = ib('sje', validator=validators.in_({'cosine', 'triplet', 'sje'}))
    swap_images     = ib(False)
    triplet_swap    = ib(False)
    cosine_margin   = ib(0.5)
    repr_loss       = ib('sje', validator=validators.in_({'euclidean', 'cosine', 'sje'}))
    device          = ib('cuda')

    ckpt_dir        = ib('/tmp')
    ckpt_prefix     = ib('CUB-SJE')

    print_freq      = ib(250)
    validation_freq = ib(2)


def main(cfg: Config):

    logger = logging.getLogger(__name__)
    th.manual_seed(cfg.seed)
    logger.info('Creating loaders...')

    cfg.log_path     = _set_log_path(cfg.log_path)
    cfg.dev_log_path = _set_log_path(cfg.dev_log_path)

    # DATA -------------------------------------------------------------------------------------------------------------

    cfg.data_dir = CubPaths(cfg.data_dir)

    # TODO(elanmart): we don't want this hardcoded here.
    D       = numpy.load(cfg.data_dir.char_lvl)
    D       = D[:, :, :cfg.doc_length].copy()

    X_train   = numpy.load(cfg.data_dir.resnet_features.train)
    Y_train   = numpy.load(cfg.data_dir.label_arrays.train)
    IDs_train = numpy.load(cfg.data_dir.index_arrays.train)

    mask     = numpy.load(cfg.data_dir.testset_mask)['unseen']
    X_test   = numpy.load(cfg.data_dir.resnet_features.test)[mask]
    Y_test   = numpy.load(cfg.data_dir.label_arrays.test)[mask]
    IDs_test = numpy.load(cfg.data_dir.index_arrays.test)[mask]

    # TODO(elanmart): once we allow models other than CNN-LSTM we need to change
    # TODO(elanmart): ... the hardcoded ``return_indices``
    train_dset = CubDataset(X=X_train,
                            Y=Y_train,
                            D=D,
                            IDs=IDs_train,
                            max_len=cfg.doc_length,
                            return_indices=True,
                            swap_image=cfg.swap_images,
                            return_negative=bool(cfg.loss == 'triplet'))

    valid_dset  = CubDataset(X=X_test,
                             Y=Y_test,
                             D=D,
                             IDs=IDs_test,
                             R=train_dset.R,
                             max_len=cfg.doc_length,
                             return_indices=True,
                             swap_image=False,
                             return_negative=False)

    train_sampler = CubBatchSampler(train_dset, cfg.batch_size)
    train_loader  = DataLoader(train_dset, batch_sampler=train_sampler, num_workers=3, pin_memory=True)
    valid_loader  = DataLoader(valid_dset, batch_size=cfg.batch_size, shuffle=True, num_workers=3, pin_memory=True)

    transfer_dset   = TransferDataset(train_dset, valid_dset, size=len(train_dset) + 1)
    transfer_loader = TransferLoader(transfer_dset, batch_size=cfg.batch_size, shuffle=True, num_workers=3, pin_memory=True)

    logger.info('Creating models.')

    # MODELS -----------------------------------------------------------------------------------------------------------

    txt_encoder = CnnRnn(alphasize=train_loader.dataset.vocab_size,
                         cnn_dim=cfg.cnn_dim,
                         emb_dim=cfg.emb_dim,
                         dropout=cfg.dropout,
                         rnn_type=cfg.rnn,
                         predictor=cfg.txt_decoder,
                         average=cfg.average).to(cfg.device)

    img_encoder = ImageEncoder(img_dim=cfg.image_dim,
                               emb_dim=cfg.emb_dim,
                               bias=cfg.img_bias,
                               n_layers=cfg.img_nlayers,
                               dropout=cfg.dropout).to(cfg.device)

    model = SJE(img_encoder=img_encoder,
                txt_encoder=txt_encoder,
                normalize_img=cfg.normalize_img,
                normalize_txt=cfg.normalize_txt).to(cfg.device)

    distinguisher = Distinguisher(emb_dim=cfg.cnn_dim,
                                  nlayers=cfg.dist_nlayers,
                                  inner_dim=cfg.dist_inner_dim,
                                  dropout=cfg.dropout).to(cfg.device)

    # TODO(elanmart): once we allow models other than CNN-LSTM we need to change
    # TODO(elanmart): ... the hardcoded ``return_indices``
    descriptions = Descriptions(full_D=D,
                                Y=Y_test,
                                IDs=IDs_test,
                                as_indices=True,
                                max_len=cfg.doc_length,
                                vocab_size=train_loader.dataset.vocab_size).to(cfg.device)

    if cfg.init_range is not None:
        for p in model.parameters():
            p.data.uniform_(-cfg.init_range, cfg.init_range)

    # OPTIMIZERS -------------------------------------------------------------------------------------------------------
    logger.info('Initializing optimizers and schdeulers...')

    params    = ChainedParams(model, distinguisher)
    optimizer = cfg.optim(params, weight_decay=cfg.decay, lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_rate)

    # CORE FUNCTIONS ---------------------------------------------------------------------------------------------------

    def training_update_function(engine: Engine, batch: Tuple[th.Tensor, th.Tensor, th.Tensor]):

        model.train()
        optimizer.zero_grad()
        rev_loss = th.tensor(0., device=cfg.device)

        # SJE update ---------------------------------------------------------------------------------------------------

        if cfg.loss == 'triplet':

            img, txt_p, txt_n = [tensor.to(cfg.device) for tensor in batch]

            fea_img   = model.img_encoder(img)
            fea_txt_p = model.txt_encoder(txt_p)
            fea_txt_n = model.txt_encoder(txt_n)

            if cfg.normalize_img:
                fea_img = F.normalize(fea_img, p=2, dim=1)

            if cfg.normalize_txt:
                fea_txt_p = F.normalize(fea_txt_p, p=2, dim=1)
                fea_txt_n = F.normalize(fea_txt_n, p=2, dim=1)

            loss = F.triplet_margin_loss(fea_img, fea_txt_p, fea_txt_n, swap=cfg.triplet_swap)

        elif cfg.loss == 'cosine':

            (img, txt, _) = batch

            img = img.to(cfg.device)
            txt = txt.to(cfg.device)

            bs   = txt.size(0)
            inds = th.cat([
                th.arange((bs // 2) - 1, -1, -1).long(),  # shuffle first half of the batch
                th.arange((bs // 2), bs, 1).long()        # retain the second one
            ])
            img  = img[inds, :]

            y_cosine = img.new_ones(img.size(0))
            y_cosine[:(bs // 2)] *= -1  # first half is shuffled

            (fea_img,
             fea_txt) = model(img, txt)

            loss = F.cosine_embedding_loss(fea_img, fea_txt, y_cosine, margin=cfg.cosine_margin)

        elif cfg.loss == 'sje':

            (img, txt, _) = batch

            img = img.to(cfg.device)
            txt = txt.to(cfg.device)

            (fea_img,
             fea_txt) = model(img, txt)

            loss = joint_embedding_loss(fea_img, fea_txt, cfg.symmetric)

        else:
            raise RuntimeError("Unknown loss")

        loss.backward()

        # GRAD-REV update ----------------------------------------------------------------------------------------------

        if cfg.rev_alpha > 0:

            # noinspection PyTypeChecker
            batch = next(transfer_loader)

            (_, txt, y) = batch
            txt = txt.to(cfg.device)
            y   = y.to(cfg.device)

            fea_txt = txt_encoder.extractor(txt)
            fea_txt = reverse_grad(fea_txt, cfg.rev_alpha)
            preds   = distinguisher(fea_txt).squeeze()

            rev_loss = F.binary_cross_entropy_with_logits(preds, y)
            rev_loss.backward()

        if cfg.grad_clip > 0:
            clip_grad_norm(model.parameters(), cfg.grad_clip)

        optimizer.step()

        return dict(
            loss     = loss,
            rev_loss = rev_loss
        )

    def validation_function(engine: Engine, batch: Tuple[th.Tensor, th.Tensor, th.Tensor]):

        with no_grad():
            model.eval()

            (img, _, y) = batch

            img   = img.to(cfg.device)
            y     = y.to(cfg.device)

            fea_img = model.img_encoder(img)

            (D_emb,
             D_emb_normalized)   = engine.state.D_emb
            (mu,
             sigma,
             mu_n,
             sigma_n) = engine.state.mu_sigma

            scores = similarity(fea_img, D_emb, mode=cfg.repr_loss)
            scores_normalized = similarity(fea_img, D_emb_normalized, mode=cfg.repr_loss)

            preds    = descriptions.predict(scores)
            preds_n  = descriptions.predict(scores_normalized)
            preds_z  = descriptions.predict(scores, mu=mu, sigma=sigma)
            preds_zn = descriptions.predict(scores_normalized, mu=mu_n, sigma=sigma_n)

            # noinspection PyUnresolvedReferences
            def _acc(p: th.Tensor) -> float:
                return float((p == y).sum()) / p.size(0)

            return dict(
                acc    = _acc(preds),
                acc_n  = _acc(preds_n),
                acc_z  = _acc(preds_z),
                acc_zn = _acc(preds_zn),
            )

    # IGNITE -------------------------------------------------------------------
    logger.info('Initializing ignite helpers ...')

    trainer        = Engine(training_update_function)
    evaluator      = Engine(validation_function)
    timer          = Timer().attach(trainer)
    train_averages = defaultdict(lambda: MovingAverage(0.99))
    test_averages  = defaultdict(Average)
    chkpointer     = ModelCheckpoint(cfg.ckpt_dir, cfg.ckpt_prefix,
                                     save_interval=1, n_saved=10, require_empty=False)

    # HELPER HOOKS -------------------------------------------------------------

    @trainer.on(Events.STARTED)
    def log(_):
        logger.info("Training Started...")

        with open(cfg.log_path, 'w'):
            pass

        with open(cfg.dev_log_path, 'w'):
            pass

    @trainer.on(Events.EPOCH_STARTED)
    def reset_timer(engine: Engine):
        timer.reset()
        scheduler.step()
        transfer_loader.reset()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_stats(engine: Engine):
        train_averages['iteraton'] = engine.state.iteration
        train_averages['epoch'] = engine.state.epoch
        train_averages['time'] = round(timer.value() / 60., 2)

        output = _filter_values(engine.state.output)
        for k, v in output.items():
            train_averages[k].update(v)

            engine.state.output = None

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_loss(engine: Engine):

        if (engine.state.iteration % cfg.print_freq) != 0:
            return

        keys = sorted(train_averages.keys())
        values = [_compute(train_averages[k]) for k in keys]

        with open(cfg.log_path, 'a+') as f:
            if f.tell() == 0:
                _write_row(keys, f)
            _write_row(values, f)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine: Engine):
        engine.state.batch  = None
        engine.state.output = None
        th.cuda.empty_cache()

        chkpointer(engine, {
            'SJE':            model,
            'Distinguisher':  distinguisher,
            'Descriptions':   descriptions
        })

        engine.state.chkpointer = chkpointer

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine: Engine, e: Exception):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            save_model(engine)
            engine.terminate()

        else:
            raise e

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine: Engine):
        if engine.state.epoch % cfg.validation_freq == 0:
            evaluator.run(valid_loader)

    @evaluator.on(Events.EPOCH_STARTED)
    def log(engine: Engine):
        logger.info("Validation started...")

    @evaluator.on(Events.EPOCH_STARTED)
    def setup(engine: Engine):
        engine.state.batch  = None
        engine.state.output = None
        th.cuda.empty_cache()

        for v in test_averages.values():
            try:
                v.reset()
            except AttributeError:
                pass

    @evaluator.on(Events.EPOCH_STARTED)
    def precompute_embeddings(engine: Engine):
        ret = descriptions.compute_representations(model=txt_encoder,
                                                   device=cfg.device,
                                                   normalize=True)
        D_emb, D_emb_normalized = ret
        engine.state.D_emb = (D_emb, D_emb_normalized)

    @evaluator.on(Events.EPOCH_STARTED)
    def update_score_stats(engine: Engine):
        with no_grad():

            model.eval()

            all_scores   = []
            all_scores_n = []

            for batch in train_loader:

                img, *_ = batch
                img     = img.to(cfg.device)

                (D_emb,
                 D_emb_normalized)   = engine.state.D_emb

                fea_img  = model.img_encoder(img)

                scores = similarity(fea_img, D_emb, mode=cfg.repr_loss)
                scores_normalized = similarity(fea_img, D_emb_normalized, mode=cfg.repr_loss)

                all_scores   += [scores]
                all_scores_n += [scores_normalized]

        all_scores   = th.cat(all_scores)
        all_scores_n = th.cat(all_scores_n)

        _mu      = all_scores.mean(   0, keepdim=True)
        _sigma   = all_scores.std(    0, keepdim=True)
        _mu_n    = all_scores_n.mean( 0, keepdim=True)
        _sigma_n = all_scores_n.std(  0, keepdim=True)

        engine.state.mu_sigma = (_mu, _sigma, _mu_n, _sigma_n)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_stats(engine: Engine):
        test_averages['epoch'] = trainer.state.epoch

        output = _filter_values(engine.state.output)
        for k, v in output.items():
            test_averages[k].update(v)

        engine.state.output = None

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_summary(engine: Engine):
        keys = sorted(test_averages.keys())
        values = [_compute(test_averages[k]) for k in keys]

        with open(cfg.dev_log_path, 'a+') as f:
            if f.tell() == 0:
                _write_row(keys, f)
            _write_row(values, f)

        engine.state.output = None

    logger.info('Kicking of trainig...')

    trainer.run(train_loader, max_epochs=cfg.max_epochs)
    trainer.model = model
    trainer.descriptions = descriptions

    return trainer


# TODO: The use of Config class doesn't play nicely with fire.Fire, hence the hack
class Runner(Config):
    def main(self):
        main(self)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelno)s -- %(asctime)s -- %(filename)s -- %(message)s')
    fire.Fire(Runner)
