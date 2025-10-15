# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, perceptual_loss=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('vis_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('mask_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('kl_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('p_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        if args.tune_decoder:
            with torch.amp.autocast('cuda'):
                loss, _, _, vis_loss, p_loss, kl_loss = model(samples, mask_ratio=args.mask_ratio)
                mask_loss = torch.zeros_like(loss)
        else:
            with torch.amp.autocast('cuda'):
                loss, _, _, vis_loss, mask_loss, kl_loss, p_loss = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        vis_loss_value = vis_loss.item()
        mask_loss_value = mask_loss.item()
        p_loss_value = p_loss.item()
        if args.kl_loss_weight is not None and not args.tune_decoder:
            kl_loss_value = kl_loss.item()
        else:
            kl_loss_value = 0.0
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(vis_loss=vis_loss_value)
        metric_logger.update(mask_loss=mask_loss_value)
        metric_logger.update(kl_loss=kl_loss_value)
        metric_logger.update(p_loss=p_loss_value)


        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        vis_loss_value_reduce = misc.all_reduce_mean(vis_loss_value)
        mask_loss_value_reduce = misc.all_reduce_mean(mask_loss_value)
        kl_loss_value_reduce = misc.all_reduce_mean(kl_loss_value)
        p_loss_value_reduce = misc.all_reduce_mean(p_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('vis_loss', vis_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('mask_loss', mask_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('kl_loss', kl_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('p_loss', p_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}