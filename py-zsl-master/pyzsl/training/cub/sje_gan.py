import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict

import numpy
import torch as th
from attr import ib, attrs
from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
from torch.autograd import no_grad
from torch.utils.data import DataLoader

from pyzsl.data.cub.paths import CubPaths
from pyzsl.models.deep.cv.gan import Generator, Discriminator, _grad_penalty
from pyzsl.models.deep.cv.similarities import similarity
from pyzsl.models.deep.cv.sje import SJE, joint_embedding_loss, ImageEncoder, CnnRnn
from pyzsl.training.cub.loaders import CubDataset, Descriptions, CubBatchSampler
from pyzsl.training.cub.sje_rev import Config
from pyzsl.utils.training import MovingAverage, Average


def _set_grad(net: th.nn.Module, flag: bool):
    for p in net.parameters():
        p.requires_grad = flag

    return net


def _filter_values(dct: Dict):
    dct = {k: float(v) for k, v in dct.items()}
    dct = {k: v for k, v in dct.items() if not math.isnan(v)}

    return dct


def _write_row(row, f):
    row = ';'.join(str(item) for item in row)
    print(row, file=f)


def _set_log_path(p):
    p = Path(p)
    assert not p.is_dir()
    p.parent.mkdir(parents=True, exist_ok=True)

    return p


def _compute(value):
    if hasattr(value, 'compute'):
        return value.compute()
    return float(value)


@attrs
class GanConfig:

    base_model_path = ib()  # type: str
    log_path        = ib()
    dev_log_path    = ib()

    gen_dim         = ib(4096)
    discr_dim       = ib(4096)
    z_dim           = ib(None)

    learning_rate_g = ib(0.001)
    learning_rate_d = ib(0.001)

    D_steps = ib(5)
    lmbda   = ib(10.)
    beta    = ib(0.01)

    transform_text      = ib(True)
    use_grad_penalty    = ib(True)
    classifier_training = ib(True)

    batch_size = ib(32)
    max_epochs = ib(10)

    print_freq      = ib(10)
    validation_freq = ib(1)

    ckpt_dir    = ib('/tmp')
    ckpt_prefix = ib('GAN_models_')


def main(cfg: Config, gan_cfg: GanConfig):

    logger = logging.getLogger(__name__)
    th.manual_seed(cfg.seed)
    logger.info('Creating loaders...')

    # prep
    gan_cfg.z_dim        = gan_cfg.z_dim or cfg.emb_dim
    gan_cfg.log_path     = _set_log_path(gan_cfg.log_path)
    gan_cfg.dev_log_path = _set_log_path(gan_cfg.dev_log_path)

    # Load the trained model
    ref_model = th.load(gan_cfg.base_model_path).to(cfg.device)  # type: SJE

    # DATA -------------------------------------------------------------------------------------------------------------

    cfg.data_dir = CubPaths(cfg.data_dir)

    D = numpy.load(cfg.data_dir.char_lvl)
    D = D[:, :, :cfg.doc_length].copy()

    X_train = numpy.load(cfg.data_dir.resnet_features.train)
    Y_train = numpy.load(cfg.data_dir.label_arrays.train)
    IDs_train = numpy.load(cfg.data_dir.index_arrays.train)

    mask = numpy.load(cfg.data_dir.testset_mask)['unseen']
    X_test = numpy.load(cfg.data_dir.resnet_features.test)[mask]
    Y_test = numpy.load(cfg.data_dir.label_arrays.test)[mask]
    IDs_test = numpy.load(cfg.data_dir.index_arrays.test)[mask]

    train_dset = CubDataset(X=X_train,
                            Y=Y_train,
                            D=D,
                            IDs=IDs_train,
                            max_len=cfg.doc_length,
                            return_indices=True,
                            swap_image=cfg.swap_images,
                            return_negative=bool(cfg.loss == 'triplet'))

    valid_dset = CubDataset(X=X_test,
                            Y=Y_test,
                            D=D,
                            IDs=IDs_test,
                            R=train_dset.R,
                            max_len=cfg.doc_length,
                            return_indices=True,
                            swap_image=False,
                            return_negative=False)

    train_sampler = CubBatchSampler(train_dset, gan_cfg.batch_size)
    train_loader = DataLoader(train_dset, batch_sampler=train_sampler, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid_dset, batch_size=gan_cfg.batch_size, shuffle=True, num_workers=3, pin_memory=True)

    logger.info('Creating models.')

    # GAN MODELS -----------------------------------------------------------------------------------------------------------

    generator = Generator(img_dim=cfg.emb_dim,
                          txt_dim=cfg.emb_dim,
                          hid_dim=gan_cfg.gen_dim,
                          z_dim=gan_cfg.z_dim).to(cfg.device)

    discriminator = Discriminator(img_dim=cfg.emb_dim,
                                  txt_dim=cfg.emb_dim,
                                  hid_dim=gan_cfg.discr_dim).to(cfg.device)

    descriptions = Descriptions(full_D=D,
                                Y=Y_test,
                                IDs=IDs_test,
                                as_indices=True,
                                max_len=cfg.doc_length,
                                vocab_size=train_loader.dataset.vocab_size).to(cfg.device)

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

    clf = SJE(img_encoder=img_encoder,
              txt_encoder=txt_encoder,
              normalize_img=cfg.normalize_img,
              normalize_txt=cfg.normalize_txt).to(cfg.device)

    optimizer_G = th.optim.Adam(
        list(generator.parameters()) + list(clf.parameters()),
        betas=(0.5, 0.99),
        lr=gan_cfg.learning_rate_g
    )

    optimizer_D = th.optim.Adam(
        discriminator.parameters(),
        betas=(0.5, 0.99),
        lr=gan_cfg.learning_rate_d
    )

    # CORE FUNCTIONS ---------------------------------------------------------------------------------------------------
    def training_update_function(engine: Engine, batch: Tuple[th.Tensor, th.Tensor, th.Tensor]):

        # Setup
        generator.to(cfg.device).train()
        discriminator.to(cfg.device).train()
        ref_model.to(cfg.device).train()
        clf.to(cfg.device).train()

        img, txt, y = [
            tensor.to(cfg.device)
            for tensor in batch
        ]

        original_txt = txt

        if gan_cfg.transform_text:
            txt = ref_model.txt_encoder(txt)

        # ----------------------
        # Update discriminator

        _set_grad(discriminator, True)
        discriminator.zero_grad()
        generator.zero_grad()

        with th.no_grad():
            fake_img = generator.sample(txt)
            real = th.cat([img, txt], dim=1)
            fake = th.cat([fake_img, txt], dim=1)

        D_real = discriminator(real).mean()
        D_fake = discriminator(fake).mean()

        gp = 0.
        if gan_cfg.use_grad_penalty:
            gp = _grad_penalty(discriminator=discriminator, real_data=real, fake_data=fake, lbd=gan_cfg.lmbda)

        D_loss = -(D_real - D_fake - gp)
        Wasserstein_D = D_real - D_fake

        D_loss.backward()
        optimizer_D.step()

        (fea_img,
         fea_txt) = ref_model(img, original_txt)

        (fake_fea_img,
         fake_fea_txt) = ref_model(fake_img, original_txt)

        D_sje_real = joint_embedding_loss(fea_img, fea_txt, symmetric=True)
        D_sje_fake = joint_embedding_loss(fake_fea_img, fake_fea_txt, symmetric=True)
        D_sje_loss = D_sje_real - D_sje_fake

        ret = dict(
            D_fake=D_fake,
            D_real=D_real,
            D_loss=D_loss,
            D_sje_fake=D_sje_fake,
            D_sje_real=D_sje_real,
            D_sje_loss=D_sje_loss,
            Wasserstein_D=Wasserstein_D,
            gp=gp,
        )

        # ----------------------
        # Update generator only once every D_steps

        if ((engine.state.iteration + 1) % gan_cfg.D_steps) != 0:
            return ret

        # ----------------------
        # Update generator

        _set_grad(discriminator, False)
        generator.zero_grad()

        fake_img = generator.sample(txt)
        fake = th.cat([fake_img, txt], dim=1)

        G = -discriminator(fake).mean()
        G.backward(retain_graph=gan_cfg.classifier_training)

        sje_loss = 'nan'
        if gan_cfg.classifier_training:
            clf.to(cfg.device).train()
            (fea_img,
             fea_txt) = clf(fake_img, original_txt)
            sje_loss = joint_embedding_loss(fea_img, fea_txt, cfg.symmetric)
            sje_loss *= gan_cfg.beta
            sje_loss.backward()

        optimizer_G.step()

        ret = {
            **ret,
            'G': G,
            'clf_loss': sje_loss
        }

        return ret

    def validation_function(engine: Engine, batch: Tuple[th.Tensor, th.Tensor, th.Tensor]):

        with no_grad():
            generator.to(cfg.device).eval()
            discriminator.to(cfg.device).eval()
            ref_model.to(cfg.device).eval()

            with no_grad():
                img, txt, y = [tensor.to(cfg.device) for tensor in batch]

                if gan_cfg.transform_text:
                    txt = ref_model.txt_encoder(txt)

                fake_img = generator.sample(txt)

                fea_img = ref_model.img_encoder(img)
                fea_fake_img = ref_model.img_encoder(fake_img)

                D_emb = engine.state.D_emb
                scores = similarity(fea_img, D_emb, mode=cfg.repr_loss)
                preds = descriptions.predict(scores)

                scores_fake = similarity(fea_fake_img, D_emb, mode=cfg.repr_loss)
                preds_fake = descriptions.predict(scores_fake)

                # noinspection PyUnresolvedReferences
                def _acc(p: th.Tensor) -> float:
                    return float((p == y).sum()) / p.size(0)

                return dict(
                    acc_preds=_acc(preds),
                    acc_fake_preds=_acc(preds_fake),
                )

    # IGNITE -----------------------------------------------------------------------------------------------------
    logger.info('Initializing ignite helpers ...')

    trainer = Engine(training_update_function)
    evaluator = Engine(validation_function)
    timer = Timer().attach(trainer)

    train_averages = defaultdict(lambda: MovingAverage(0.99))
    test_averages = defaultdict(Average)

    chkpointer = ModelCheckpoint(gan_cfg.ckpt_dir, gan_cfg.ckpt_prefix,
                                 save_interval=1, n_saved=10, require_empty=False)

    # HELPER HOOKS -----------------------------------------------------------------------------------------------------

    @trainer.on(Events.STARTED)
    def log(_):
        logger.info("Training Started...")

        with open(gan_cfg.log_path, 'w'):
            pass

        with open(gan_cfg.dev_log_path, 'w'):
            pass

    @trainer.on(Events.EPOCH_STARTED)
    def reset_timer(_):
        timer.reset()

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

        if (engine.state.iteration % gan_cfg.print_freq) != 0:
            return

        keys = sorted(train_averages.keys())
        values = [_compute(train_averages[k]) for k in keys]

        with open(gan_cfg.log_path, 'a+') as f:
            if f.tell() == 0:
                _write_row(keys, f)
            _write_row(values, f)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine: Engine):
        engine.state.batch = None
        engine.state.output = None
        th.cuda.empty_cache()

        chkpointer(engine, {
            'generator': generator.to('cpu').eval(),
            'discriminator': discriminator.to('cpu').eval(),
            'classifier': clf.to('cpu').eval(),
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
        if engine.state.epoch % gan_cfg.validation_freq == 0:
            evaluator.run(valid_loader)

    @evaluator.on(Events.EPOCH_STARTED)
    def log(_):
        logger.info("Validation started...")

    @evaluator.on(Events.EPOCH_STARTED)
    def setup(engine: Engine):
        engine.state.batch = None
        engine.state.output = None
        th.cuda.empty_cache()

        for v in test_averages.values():
            try:
                v.reset()
            except AttributeError:
                pass

    @evaluator.on(Events.EPOCH_STARTED)
    def precompute_embeddings(engine: Engine):

        ref_model.txt_encoder.to(cfg.device).eval()
        descriptions.to(cfg.device)

        ret = descriptions.compute_representations(model=ref_model.txt_encoder,
                                                   device=cfg.device,
                                                   normalize=True)

        D_emb, _ = ret
        engine.state.D_emb = D_emb

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_stats(engine: Engine):
        test_averages['epoch'] = engine.state.epoch

        output = _filter_values(engine.state.output)
        for k, v in output.items():
            test_averages[k].update(v)

        engine.state.output = None

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_summary(engine: Engine):
        keys = sorted(test_averages.keys())
        values = [_compute(test_averages[k]) for k in keys]

        with open(gan_cfg.dev_log_path, 'a+') as f:
            if f.tell() == 0:
                _write_row(keys, f)
            _write_row(values, f)

        engine.state.output = None

    # TRAINING ---------------------------------------------------------------------------------------------------------
    logger.info('Kicking of trainig...')

    trainer.run(train_loader, max_epochs=gan_cfg.max_epochs)

    return trainer
