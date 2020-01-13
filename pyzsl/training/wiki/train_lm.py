import logging
from typing import Tuple

import numpy as np
import torch
import torch as th
from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
from ignite.metrics import CategoricalAccuracy
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from pyzsl.models.deep.lm.data import LMLoader
from pyzsl.models.deep.lm.rnn import RNNModel
from pyzsl.models.deep.lm.tcn import TCNModel
from pyzsl.models.deep.lm.utils import AdaptiveLogSoftmaxWithLoss, \
    STLRScheduler, \
    get_discriminative_optimizer, set_weight
from pyzsl.utils.general import ChainedParams
from pyzsl.utils.training import Average, MovingAverage, detach, Unfreezer

logger = logging.getLogger(__name__)
FREQ   = 1000


def main(train: np.ndarray,
         dev: np.ndarray,
         nn_type: str,
         ninp: int, nhid: int, nlayers: int,
         dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.5, wdrop=0.0,
         tie_weights=True,
         bsz=64, bptt=70, max_epochs=10, eval_bsz=None, eval_bptt=None,
         lr=0.01, clip=0.0,
         adaptive=False, cutoffs=None,
         device='cuda', to_device=False,
         ckpt_embedding=True,
         lock_emb=True,
         kernel_size=4, num_levels=5,
         finetune=False, reverse_layers=False,
         module=None, decoder=None,
         ckpt_dir='/tmp/', ckpt_prefix='LM-model',
         seed=0):

    torch.manual_seed(seed)

    logger.debug('Creating data loaders...')
    eval_bsz  = eval_bsz or bsz
    eval_bptt = eval_bptt or bptt
    trn_dset  = LMLoader(train, device=device, bptt=bptt, batch_size=bsz, evaluation=False, to_device=to_device)
    val_dset  = LMLoader(dev,   device=device, bptt=eval_bptt, batch_size=eval_bsz, evaluation=True, to_device=to_device)

    ntokens = max(int(train.max()), int(dev.max())) + 1
    nout    = ninp if tie_weights else nhid

    logger.debug(f'Data lengths: {len(trn_dset)}, {len(val_dset)}. Ntokens: {ntokens}')
    logger.debug(f'Creating model...')

    if module is None:

        if nn_type in {'LSTM', 'QRNN'}:
            module = RNNModel(
                rnn_type=nn_type, ntoken=ntokens, ninp=ninp, nhid=nhid, nlayers=nlayers,
                dropout=dropout, dropouth=dropouth, dropouti=dropouti, dropoute=dropoute,
                wdrop=wdrop, tie_weights=tie_weights, ckpt_embedding=ckpt_embedding,
                cuda=(device.startswith('cuda'))
            )

        elif nn_type in {'TCN'}:
            module = TCNModel(input_size=ninp, output_size=ntokens,
                              kernel_size=kernel_size, num_channels=[nhid] * (num_levels - 1) + [nhid],
                              lock_emb=lock_emb, dropout=dropout, emb_dropout=dropoute)

        else:
            raise ValueError(f'Unrecognized NN type: {nn_type}')

    if adaptive:
        if decoder is None:
            logger.debug(f'Creating adaptive decoder for cutoffs {cutoffs}...')
            decoder = AdaptiveLogSoftmaxWithLoss(in_features=nout, n_classes=ntokens, cutoffs=cutoffs, head_bias=(not tie_weights))
        loss_fn = lambda input, target: decoder(input, target).loss

    else:
        if decoder is None:
            logger.debug('Creating standard decoder...')
            decoder = nn.Linear(in_features=nout, out_features=ntokens, bias=(not tie_weights))
        loss_fn = lambda input, target: F.cross_entropy(decoder(input), target)

    if tie_weights and (not finetune):  # when we finetune we do gradual unfreezing, therefore we cannot tie weights
        set_weight(decoder, module.encoder.weight)

    def traininig_step(engine: Engine, batch: Tuple[th.Tensor, th.Tensor]):
        x, y = batch  # x: (L, B);  y: (L, )

        module.train()
        decoder.train()
        optimizer.zero_grad()
        engine.state.hidden = detach(engine.state.hidden)

        (output,  # (L, B, n_hidden) or (L, B, n_input)
         engine.state.hidden) = module(x, engine.state.hidden)

        output = output.view(-1, output.size(2))

        loss = loss_fn(output, y)  # type: th.Tensor
        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)

        optimizer.step()
        trn_loss.update(loss.item())

        # saving that tiny bit of memory
        engine.state.batch = None
        engine.state.output = None

    def validation_step(engine: Engine, batch: Tuple[th.Tensor, th.Tensor]):
        with th.no_grad():
            x, y = batch

            module.eval()
            decoder.eval()
            engine.state.hidden = detach(engine.state.hidden)

            (output,
             engine.state.hidden) = module(x, engine.state.hidden)

            output  = output.view(-1, output.size(2))
            n_items = output.shape[0] * 1.0

            if adaptive:
                output = decoder.log_prob(output)
                loss   = F.nll_loss(output, y)
            else:
                output = decoder(output)
                loss   = F.cross_entropy(output, y)

            val_loss.update(loss.item())
            accuarcy.update((output, y))

            # saving that tiny bit of memory
            engine.state.batch  = None
            engine.state.output = None

            return {
                'n_items': n_items,
                'loss': loss.item(),
            }

    module     = module.to(device)
    decoder    = decoder.to(device)
    params     = ChainedParams(module, decoder)
    optimizer  = th.optim.Adam(params, lr=lr, amsgrad=True)
    trainer    = Engine(traininig_step)
    validator  = Engine(validation_step)
    timer      = Timer().attach(trainer)
    trn_loss   = MovingAverage(0.99)
    val_loss   = Average()
    accuarcy   = CategoricalAccuracy()
    chkpointer = ModelCheckpoint(ckpt_dir, ckpt_prefix,
                                 save_interval=1, n_saved=10, require_empty=False)

    logger.debug(f'Helpers are ready...')
    logger.info(f'Total of {params.size():_} parameters will be optimized')
    logger.info(f'{ChainedParams(module).size():_} model parameters '
                f'and {ChainedParams(decoder).size():_} decoder parameters')

    if finetune:
        logger.info("Finetuning mode is set to TRUE. Prepping helpers...")

        layers    = module.get_layers() + [decoder]
        layers_p  = [list(ChainedParams(layer)) for layer in layers]
        defreezer = Unfreezer(layers=layers, reverse=reverse_layers)
        optimizer = get_discriminative_optimizer(cls=Adam, layered_params=layers_p, base_lr=lr, discount_factor=2.6)
        scheduler = STLRScheduler(optimizer=optimizer, T=(max_epochs * len(trn_dset)))

        trainer.add_event_handler(Events.ITERATION_STARTED, lambda _: scheduler.step())
        trainer.add_event_handler(Events.EPOCH_STARTED,     lambda _: defreezer.step())

    @trainer.on(Events.EPOCH_STARTED)
    def reset_timer(engine: Engine):
        timer.reset()
        trn_loss.reset()
        module.reset()

        engine.state.hidden = module.init_hidden(bsz)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_loss(engine: Engine):
        if (engine.state.iteration % FREQ) == 0:
            it  = engine.state.iteration
            ls  = trn_loss.compute()
            msg = f'Iteration no {it} / {len(trn_dset)}, loss: {ls}'
            logger.info(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine: Engine):
        engine.state.batch  = None
        engine.state.hidden = None
        engine.state.output = None

        th.cuda.empty_cache()

        chkpointer(engine, {
            'module':  module,
            'decoder': decoder
        })

        engine.state.chkpointer = chkpointer

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_summary(engine: Engine):
        ts = round(timer.value() / 60., 2)
        engine.state.metrics['loss'] = trn_loss.compute()
        ls = engine.state.metrics['loss']

        logger.info(
            f' --------------------------------- '
            f'Epoch {engine.state.epoch} done.   '
            f'Time elapsed: {ts:.3f}[min]        '
            f'Average loss: {ls}                 '
            f'Triggering evaluation run...       '
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(_):
        validator.run(val_dset)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine: Engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            chkpointer(engine, {
                'module':  module,
                'decoder': decoder
            })

            engine.terminate()

        else:
            raise e

    @validator.on(Events.EPOCH_STARTED)
    def setup(engine: Engine):
        engine.state.batch  = None
        engine.state.hidden = None
        engine.state.output = None

        th.cuda.empty_cache()

        val_loss.reset()
        accuarcy.reset()
        module.reset()

        engine.state.hidden = module.init_hidden(bsz)

    @validator.on(Events.EPOCH_COMPLETED)
    def print_summary(engine: Engine):
        ls = engine.state.metrics['loss']     = val_loss.compute()
        ac = engine.state.metrics['accuracy'] = accuarcy.compute()

        logger.info(
            f'~~~ VALIDATION ~~~~~~~~~~~~~~~~~ '
            f'Average loss:     {ls:.2f}        '
            f'Average accuracy: {ac:.2f}        '
        )

    logger.debug(f'Invoking trainer.run for {max_epochs} epochs')
    return trainer.run(data=trn_dset, max_epochs=max_epochs)
