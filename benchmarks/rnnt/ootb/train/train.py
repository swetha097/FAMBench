# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license

import argparse
import copy
import os
import random
import time
import sys

import torch
import numpy as np
import torch.distributed as dist
from apex import amp
from apex.optimizers import FusedLAMB
from apex.parallel import DistributedDataParallel

from common import helpers
from common.data.text import Tokenizer
from common.data import features
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch)
from common.optimizers import lr_policy
from common.tb_dllogger import flush_log, init_log, log
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.loss import RNNTLoss
from rnnt.model import RNNT
import torch.multiprocessing as mp
from mlperf import logging

# FB5 Logger
import pathlib
p = pathlib.Path(__file__).parent.resolve() / "../../../../fb5logging"
sys.path.append(os.fspath(p))
from fb5logger import FB5Logger
import loggerconstants


def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=100, type=int,
                          help='Number of epochs for the entire training')
    training.add_argument("--warmup_epochs", default=6, type=int,
                          help='Initial epochs of increasing learning rate')
    training.add_argument("--hold_epochs", default=40, type=int,
                          help='Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int,
                          help=('Run for a number of epochs with no effect on the lr schedule.'
                                'Useful for re-starting the training.'))
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp', '--fp16', action='store_true', default=False,
                          help='Use mixed precision training')
    training.add_argument('--seed', default=None, type=int, help='Random seed')
    training.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                          help='GPU id used for distributed training')
    training.add_argument('--target', default=0.058, type=float, help='Target WER accuracy')
    training.add_argument('--weights_init_scale', default=0.5, type=float, help='If set, overwrites value in config.')
    training.add_argument('--hidden_hidden_bias_scale', type=float, help='If set, overwrites value in config.')

    optim = parser.add_argument_group('optimization setup')
    #optim.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')

    optim.add_argument('--batch_size', default=128, type=int,
                       help='Effective batch size per GPU (might require grad accumulation')
    optim.add_argument('--val_batch_size', default=2, type=int,
                       help='Evalution time batch size')
    optim.add_argument('--lr', default=4e-3, type=float,
                       help='Peak learning rate')
    optim.add_argument("--min_lr", default=1e-5, type=float,
                       help='minimum learning rate')
    optim.add_argument("--lr_exp_gamma", default=0.935, type=float,
                       help='gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=1e-3, type=float,
                       help='Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=8, type=int,
                       help='Number of accumulation steps')
    optim.add_argument('--log_norm', action='store_true',
                       help='If enabled, gradient norms will be logged')
    optim.add_argument('--clip_norm', default=1, type=float,
                       help='If provided, gradients will be clipped above this norm')
    optim.add_argument('--beta1', default=0.9, type=float, help='Beta 1 for optimizer')
    optim.add_argument('--beta2', default=0.999, type=float, help='Beta 2 for optimizer')
    optim.add_argument('--ema', type=float, default=0.999,
                       help='Discount factor for exp averaging of model weights')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--nodali', action='store_true', default=True,
                    help='Don\'t use DALI pipeline for fast data processing')
    io.add_argument('--device', type=str, choices=['cpu', 'gpu'],
                    default='gpu', help='Use device for execution.')
    io.add_argument('--num-workers', default=6, type=int,
                    help='Number of workers used in data-loading')
    io.add_argument('--resume', action='store_true',
                    help='Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to a checkpoint for resuming training')
    io.add_argument('--save_at_the_end', action='store_true',
                    help='Saves model checkpoint at the end of training')
    io.add_argument('--save_frequency', default=None, type=int,
                    help='Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[], type=int, nargs='+',
                    help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=200, type=int,
                    help='Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--val_frequency', default=1, type=int,
                    help='Number of epochs between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int,
                    help='Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=None, type=int,
                    help='Number of steps between printing sample decodings')
    io.add_argument('--model_config', default='configs/baseline_v3-1023sp.yaml',
                    type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--num_buckets', type=int, default=6,
                    help='If provided, samples will be grouped by audio duration, '
                         'to this number of backets, for each bucket, '
                         'random samples are batched, and finally '
                         'all batches are randomly shuffled')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
                    help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--max_duration', type=float,
                    help='Discard samples longer than max_duration')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True,
                    help='Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to save the training logfile.')
    io.add_argument('--max_symbol_per_sample', type=int, default=None,
                    help='maximum number of symbols per sample can have during eval')
    io.add_argument('--mlperf', action='store_true', help='Enable MLPerf Logging.')

    # FB5 Logging
    io.add_argument("--fb5logger", type=str, default=None)
    io.add_argument("--fb5config", type=str, default="small")
    return parser.parse_args()


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = getattr(model, 'module', model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])


@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize,
             ema_model, loss_fn, greedy_decoder, use_amp, args):

    ema_model.eval()

    start_time = time.time()
    agg = {'losses': [], 'preds': [], 'txts': [], 'idx': []}
    if args.mlperf:
        logging.log_start(logging.constants.EVAL_START, metadata=dict(epoch_num=epoch))
    for i, batch in enumerate(val_loader):
        print(f'{val_loader.pipeline_type} evaluation: {i:>10}/{len(val_loader):<10}', end='\r')

        audio, audio_lens, txt, txt_lens = batch

        feats, feat_lens = val_feat_proc([audio, audio_lens])

        log_probs, log_prob_lens = ema_model(feats, feat_lens, txt, txt_lens)
        loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                       log_prob_lens, txt, txt_lens)

        pred = greedy_decoder.decode(ema_model, feats, feat_lens)

        agg['losses'] += helpers.gather_losses([loss.cpu()])
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts'] += helpers.gather_transcripts([txt.cpu()], [txt_lens.cpu()], detokenize)

    wer, loss = process_evaluation_epoch(agg)
    if args.mlperf:
        logging.log_event(logging.constants.EVAL_ACCURACY, value=wer, metadata=dict(epoch_num=epoch))
        logging.log_end(logging.constants.EVAL_STOP, metadata=dict(epoch_num=epoch))

    log((epoch,), step, 'dev_ema', {'loss': loss, 'wer': 100.0 * wer, 'took': time.time() - start_time})
    ema_model.train()
    return wer


def main():

    args = parse_args()

    if args.mlperf:
        logging.configure_logger('RNNT')
        logging.log_start(logging.constants.INIT_START)

    if args.fb5logger is not None:
        fb5logger = FB5Logger(args.fb5logger)
        fb5logger.header("RNN-T", "OOTB", "train", args.fb5config, score_metric=loggerconstants.EXPS)

    #assert(torch.cuda.is_available())
    assert args.prediction_frequency is None or args.prediction_frequency % args.log_frequency == 0

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    mp.set_start_method("spawn")

    # set up distributed training
    multi_gpu = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1

    if args.seed is not None:
        if args.mlperf:
            logging.log_event(logging.constants.SEED, value=args.seed)
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        # np_rng is used for buckets generation, and needs the same seed on every worker
        np_rng = np.random.default_rng(seed=args.seed)

    init_log(args)

    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)

    assert args.grad_accumulation_steps >= 1
    assert args.batch_size % args.grad_accumulation_steps == 0, \
        f'{args.batch_size} % {args.grad_accumulation_steps} != 0'

    batch_size = args.batch_size // args.grad_accumulation_steps
    if args.mlperf:
        logging.log_event(logging.constants.GRADIENT_ACCUMULATION_STEPS, value=args.grad_accumulation_steps)
        logging.log_event(logging.constants.SUBMISSION_BENCHMARK, value=logging.constants.RNNT)
        logging.log_event(logging.constants.SUBMISSION_ORG, value='my-organization')
        logging.log_event(logging.constants.SUBMISSION_DIVISION, value=logging.constants.CLOSED)  # closed or open
        logging.log_event(logging.constants.SUBMISSION_STATUS, value=logging.constants.ONPREM)  # on-prem/cloud/research
        logging.log_event(logging.constants.SUBMISSION_PLATFORM, value='my platform')
        logging.log_end(logging.constants.INIT_STOP)

    if multi_gpu:
        torch.distributed.barrier()
    if args.mlperf:
        logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        torch.distributed.barrier()

    print_once('Setting up datasets...')
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_specaugm_kw,
    ) = config.input(cfg, 'train')
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, 'val')
    if args.mlperf:
        logging.log_event(logging.constants.DATA_TRAIN_MAX_DURATION,
                          value=train_dataset_kw['max_duration'])
        logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MAX,
                          value=train_dataset_kw['speed_perturbation']['max_rate'])
        logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MIN,
                          value=train_dataset_kw['speed_perturbation']['min_rate'])
        logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_N,
                          value=train_specaugm_kw['freq_masks'])
        logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MIN,
                          value=train_specaugm_kw['min_freq'])
        logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MAX,
                          value=train_specaugm_kw['max_freq'])
        logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_N,
                          value=train_specaugm_kw['time_masks'])
        logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MIN,
                          value=train_specaugm_kw['min_time'])
        logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MAX,
                          value=train_specaugm_kw['max_time'])
        logging.log_event(logging.constants.GLOBAL_BATCH_SIZE,
                          value=batch_size * world_size * args.grad_accumulation_steps)

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)


    class PermuteAudio(torch.nn.Module):
        def forward(self, x):
            return (x[0].permute(2, 0, 1), *x[1:])

    train_augmentations = torch.nn.Sequential(
        train_specaugm_kw and features.SpecAugment(optim_level=args.amp, **train_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **train_splicing_kw),
        PermuteAudio(),
    )
    val_augmentations = torch.nn.Sequential(
        val_specaugm_kw and features.SpecAugment(optim_level=args.amp, **val_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **val_splicing_kw),
        PermuteAudio(),
    )

    if args.mlperf:
        logging.log_event(logging.constants.DATA_TRAIN_NUM_BUCKETS, value=args.num_buckets)

    if not args.nodali:
        from common.data.dali import sampler as dali_sampler
        from common.data.dali.data_loader import DaliDataLoader

        if args.num_buckets is not None:
            sampler = dali_sampler.BucketingSampler(
                args.num_buckets,
                batch_size,
                world_size,
                args.epochs,
                np_rng
            )
        else:
            sampler = dali_sampler.SimpleSampler()

        train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                      dataset_path=args.dataset_dir,
                                      config_data=train_dataset_kw,
                                      config_features=train_features_kw,
                                      json_names=args.train_manifests,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      grad_accumulation_steps=args.grad_accumulation_steps,
                                      pipeline_type="train",
                                      device_type=args.device,
                                      tokenizer=tokenizer)

        val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                    dataset_path=args.dataset_dir,
                                    config_data=val_dataset_kw,
                                    config_features=val_features_kw,
                                    json_names=args.val_manifests,
                                    batch_size=args.val_batch_size,
                                    sampler=dali_sampler.SimpleSampler(),
                                    pipeline_type="val",
                                    device_type=args.device,
                                    tokenizer=tokenizer)
    else:
        from common.data.data_loader import AudioDataLoader

        #TODO, pass in config data for sample_rate, etc. potentially use wrapper class to drive dataset, sampler, and dataloader creation, returning just the dataloader
        train_loader = AudioDataLoader(
            config_features=train_features_kw,
            pipeline_type="train",
            gpu_id=args.local_rank,
            data_dir=args.dataset_dir,
            tokenizer=tokenizer,
            manifest_fpaths=args.train_manifests,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=args.local_rank,
            num_workers=args.num_workers,
            device_type=args.device)

        val_loader = AudioDataLoader(
            config_features=val_features_kw,
            pipeline_type="val",
            gpu_id=args.local_rank,
            data_dir=args.dataset_dir,
            tokenizer=tokenizer,
            manifest_fpaths=args.val_manifests,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=args.local_rank,
            num_workers=args.num_workers,
            device_type=args.device)

    train_feat_proc = train_augmentations
    val_feat_proc = val_augmentations

    train_feat_proc.cuda()
    val_feat_proc.cuda()

    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    if args.mlperf:
        logging.log_event(logging.constants.TRAIN_SAMPLES, value=train_loader.dataset_size)
        logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.dataset_size)

    # set up the model
    rnnt_config = config.rnnt(cfg)
    rnnt_config['mlperf'] = args.mlperf
    if args.mlperf:
        logging.log_event(logging.constants.MODEL_WEIGHTS_INITIALIZATION_SCALE, value=args.weights_init_scale)
    if args.weights_init_scale is not None:
        rnnt_config['weights_init_scale'] = args.weights_init_scale
    if args.hidden_hidden_bias_scale is not None:
        rnnt_config['hidden_hidden_bias_scale'] = args.hidden_hidden_bias_scale
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    model.cuda()
    blank_idx = tokenizer.num_labels
    loss_fn = RNNTLoss(blank_idx=blank_idx)
    if args.mlperf:
        logging.log_event(logging.constants.EVAL_MAX_PREDICTION_SYMBOLS, value=args.max_symbol_per_sample)
    greedy_decoder = RNNTGreedyDecoder(blank_idx=blank_idx,
                                       max_symbol_per_sample=args.max_symbol_per_sample)

    print_once(f'Model size: {num_weights(model) / 10**6:.1f}M params\n')

    opt_eps = 1e-9
    if args.mlperf:
        logging.log_event(logging.constants.OPT_NAME, value='lamb')
        logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)
        logging.log_event(logging.constants.OPT_LAMB_EPSILON, value=opt_eps)
        logging.log_event(logging.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=args.lr_exp_gamma)
        logging.log_event(logging.constants.OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
        logging.log_event(logging.constants.OPT_LAMB_LR_HOLD_EPOCHS, value=args.hold_epochs)
        logging.log_event(logging.constants.OPT_LAMB_BETA_1, value=args.beta1)
        logging.log_event(logging.constants.OPT_LAMB_BETA_2, value=args.beta2)
        logging.log_event(logging.constants.OPT_GRADIENT_CLIP_NORM, value=args.clip_norm)
        logging.log_event(logging.constants.OPT_LR_ALT_DECAY_FUNC, value=True)
        logging.log_event(logging.constants.OPT_LR_ALT_WARMUP_FUNC, value=True)
        logging.log_event(logging.constants.OPT_LAMB_LR_MIN, value=args.min_lr)
        logging.log_event(logging.constants.OPT_WEIGHT_DECAY, value=args.weight_decay)

    # optimization
    kw = {'params': model.param_groups(args.lr), 'lr': args.lr,
          'weight_decay': args.weight_decay}

    initial_lrs = [group['lr'] for group in kw['params']]

    print_once(f'Starting with LRs: {initial_lrs}')
    optimizer = FusedLAMB(betas=(args.beta1, args.beta2), eps=opt_eps, max_grad_norm=args.clip_norm, **kw)

    adjust_lr = lambda step, epoch: lr_policy(
        step, epoch, initial_lrs, optimizer, steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs, hold_epochs=args.hold_epochs,
        min_lr=args.min_lr, exp_gamma=args.lr_exp_gamma)

    if args.amp:
        model, optimizer = amp.initialize(
            models=model,
            optimizers=optimizer,
            opt_level='O1',
            max_loss_scale=512.0)

    if args.ema > 0:
        ema_model = copy.deepcopy(model).cuda()
    else:
        ema_model = None
    if args.mlperf:
        logging.log_event(logging.constants.MODEL_EVAL_EMA_FACTOR, value=args.ema)

    if multi_gpu:
        model = DistributedDataParallel(model)

    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T',
                                args.keep_milestones, args.amp)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)

    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    last_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # FB5 Log for a certain amount of time.
    if args.fb5logger is not None:
        fb5logger.run_start()
    total_batches = 0
    start_time = time.time()
    MAX_TIME = 120.0
    # Start Batch Loop

    # training loop
    model.train()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if args.mlperf:
            logging.log_start(logging.constants.BLOCK_START,
                              metadata=dict(first_epoch_num=epoch,
                                            epoch_count=1))
            logging.log_start(logging.constants.EPOCH_START,
                              metadata=dict(epoch_num=epoch))

        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()

        for batch in train_loader:
            if accumulated_batches == 0:
                adjust_lr(step, epoch)
                optimizer.zero_grad()
                step_utts = 0
                step_start_time = time.time()
                all_feat_lens = []

            audio, audio_lens, txt, txt_lens = batch
            #TODO is this the best place to move tensors over to cuda?  Does everything need to go? maybe batch.cuda?
            #Our initial spectrogram and mel filter aug's aren't on gpu's it seems, we should look into this.
            # audio = audio.cuda()
            # audio_lens = audio_lens.cuda()
            # txt = txt.cuda()
            # txt_lens = txt_lens.cuda()

            feats, feat_lens = train_feat_proc([audio, audio_lens])
            all_feat_lens += feat_lens
            log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens)
            loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                           log_prob_lens, txt, txt_lens)

            loss /= args.grad_accumulation_steps

            del log_probs, log_prob_lens

            if torch.isnan(loss).any():
                print_once('WARNING: loss is NaN; skipping update')
            else:
                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss_item = loss.item()
                del loss
                step_utts += batch[0].size(0) * world_size
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1
                total_batches += 1

            if accumulated_batches % args.grad_accumulation_steps == 0:

                total_norm = 0.0

                try:
                    if args.log_norm:
                        for p in getattr(model, 'module', model).parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                except AttributeError as e:
                    print_once(f'Exception happened: {e}')
                    total_norm = 0.0

                optimizer.step()
                apply_ema(model, ema_model, args.ema)

                if step % args.log_frequency == 0 or (time.time() - start_time) > MAX_TIME:

                    if args.prediction_frequency is None or step % args.prediction_frequency == 0:
                        preds = greedy_decoder.decode(model, feats, feat_lens)
                        wer, pred_utt, ref = greedy_wer(preds,
                                                        txt,
                                                        txt_lens,
                                                        tokenizer.detokenize)
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')
                        wer = {'wer': 100 * wer}
                    else:
                        wer = {}

                    step_time = time.time() - step_start_time

                    log((epoch, step % steps_per_epoch or steps_per_epoch, steps_per_epoch),
                        step, 'train',
                        {'loss': loss_item,
                         **wer,  # optional entry
                         'throughput': step_utts / step_time,
                         'took': step_time,
                         'grad-norm': total_norm,
                         'seq-len-min': min(all_feat_lens).item(),
                         'seq-len-max': max(all_feat_lens).item(),
                         'lrate': optimizer.param_groups[0]['lr']})

                    # FB5 Logger
                    if (time.time() - start_time) > MAX_TIME:
                        break

                step_start_time = time.time()

                step += 1
                accumulated_batches = 0
                # end of step
        if args.mlperf:
            logging.log_end(logging.constants.EPOCH_STOP,
                            metadata=dict(epoch_num=epoch))

        epoch_time = time.time() - epoch_start_time
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts / epoch_time,
                                          'took': epoch_time})

        # FB5 Logger
        if (time.time() - start_time) > MAX_TIME:
            break

        if epoch % args.val_frequency == 0:
            wer = evaluate(epoch, step, val_loader, val_feat_proc,
                tokenizer.detokenize, ema_model, loss_fn,
                greedy_decoder, args.amp, args)

            last_wer = wer
            if wer < best_wer and epoch >= args.save_best_from:
               checkpointer.save(model, ema_model, optimizer, epoch,
                                 step, best_wer, is_best=True)
               best_wer = wer

        save_this_epoch = (args.save_frequency is not None and epoch % args.save_frequency == 0) \
            or (epoch in args.keep_milestones)
        if save_this_epoch:
            checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)
        if args.mlperf:
            logging.log_end(logging.constants.BLOCK_STOP, metadata=dict(first_epoch_num=epoch))

        if last_wer <= args.target:
            if args.mlperf:
                logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'success'})
            if args.fb5logger is not None:
                fb5logger.run_stop(total_batches, args.batch_size * world_size)
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        # end of epoch

    log((), None, 'train_avg', {'throughput': epoch_utts / epoch_time})

    if last_wer > args.target:
        if args.mlperf:
            logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'aborted'})
        if args.fb5logger is not None:
            fb5logger.run_stop(total_batches, args.batch_size * world_size)

    if epoch == args.epochs:
        evaluate(epoch, step, val_loader, val_feat_proc, tokenizer.detokenize,
                 ema_model, loss_fn, greedy_decoder, args.amp, args)

    flush_log()
    if args.save_at_the_end:
        checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)


if __name__ == "__main__":
    main()
