#!/usr/bin/python3

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from apex import amp

import sys
import os
from tqdm import tqdm
import time
import datetime
import argparse
import numpy as np
import logging

from base.utils.log import set_logger
from base.utils.metric_logger import MetricLogger
from base.utils.checkpoint import Checkpointer
from base.solver import make_optimizer, make_scheduler, make_warmup_scheduler
from base.net.baseline import BaseNet
from base.data.fashion import FashionDataset
from base.config import defaults as df

def train(args):
    logger = logging.getLogger("main.trainer")
    try:
        model = BaseNet(
            margin=args.margin, 
            omega=args.omega, 
            use_hardtriplet=args.use_hardtriplet,
            use_labelsmooth=args.use_labelsmooth
        )
        model.to(args.device)
    except Exception as e:
        logger.error("Initialize error: {}".format(e))
        return
    logger.info("Training {}.".format(df.MODEL_NAME))
    writer = SummaryWriter(log_dir=args.out_dir, flush_secs=10)
    rand_x = torch.rand(args.batch_size, 3, *df.INPUT_SIZE)
    rand_x = rand_x.to(args.device)
    writer.add_graph(model, rand_x, False)

    optimizer = make_optimizer(args, model)
    if args.use_warmup:
        scheduler = make_warmup_scheduler(args, optimizer)
    else:
        scheduler = make_scheduler(args, optimizer)

    if args.device != torch.device("cpu"):
        amp_opt_level = 'O1' if args.use_amp else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    arguments = {}
    arguments.update(vars(args))
    arguments["itr"] = 0
    checkpointer = Checkpointer(model, 
                                optimizer=optimizer, 
                                scheduler=scheduler,
                                save_dir=args.out_dir, 
                                save_to_disk=True)
    
    batch_size = args.batch_size
    fashion = FashionDataset(item_num=args.iteration_num*batch_size)
    dataloader = DataLoader(dataset=fashion, shuffle=True, num_workers=8, batch_size=batch_size)

    model.train()
    meters = MetricLogger(delimiter=", ")
    max_itr = len(dataloader)
    start_itr = arguments["itr"] + 1
    itr_start_time = time.time()
    training_start_time = time.time()
    for itr, batch_data in enumerate(dataloader, start_itr):
        batch_data = (bd.to(args.device) for bd in batch_data)
        loss_dict = model.loss(*batch_data)
        optimizer.zero_grad()
        if args.device != torch.device("cpu"):
            with amp.scale_loss(loss_dict["loss"], optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            loss_dict["loss"].backward()
        optimizer.step()
        scheduler.step()

        arguments["itr"] = itr
        meters.update(**loss_dict)
        itr_time = time.time() - itr_start_time
        itr_start_time = time.time()
        meters.update(itr_time=itr_time)
        if itr % 50 == 0:
            eta_seconds = meters.itr_time.avg * (max_itr - itr)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            lr = optimizer.param_groups[0]["lr"] ## lr = scheduler.get_lr()[0]
            logger.info(
                meters.delimiter.join(
                    [
                        "itr: {itr}/{max_itr}",
                        "lr: {lr:.7f}",
                        "{meters}",
                        "eta: {eta}\n",
                    ]
                ).format(
                    itr=itr,
                    lr=lr,
                    max_itr=max_itr,
                    meters=str(meters),
                    eta=eta,
                )
            )
            writer.add_scalar("loss/total_loss", meters.loss.avg, itr)
            writer.add_scalar("lr", lr, itr)
            writer.add_scalar("loss/triplet_loss", meters.triplet_loss.avg, itr)
            writer.add_scalar("loss/xentropy_loss", meters.xentropy_loss.avg, itr)

        ## save model
        if itr % args.checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(itr), **arguments)
        if itr == max_itr:
            checkpointer.save("model_final", **arguments)
            break

    training_time = time.time() - training_start_time
    training_time = str(datetime.timedelta(seconds=int(training_time)))
    logger.info("total training time: {}".format(training_time))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
    parser.add_argument('--not_use_gpu', dest='not_use_gpu', action='store_true', help='do not use gpu')
    parser.add_argument('--iteration_num', type=int, default=25000, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--checkpoint_period', type=int, default=10000, help='period to save checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[12000, 20000], help='milestones for lr_scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for lr_scheduler')
    parser.add_argument('--margin', type=float, default=0.3, help='margin')
    parser.add_argument('--omega', type=float, default=0.9, help='weight between losses')
    parser.add_argument('--use-labelsmooth', dest='use_labelsmooth', action='store_true', help='use CrossEntropyLabelSmooth, otherwise CrossEntropyLoss')
    parser.add_argument('--use-warmup', dest='use_warmup', action='store_true', help='use warmup-scheduler for training')
    parser.add_argument('--warmup_factor', type=float, default=0.01, help='warmup factor for initial low-lr training')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup training iterations')
    parser.add_argument('--warmup_method', type=str, default="linear", help=' method to increase warmup lr')
    parser.add_argument('--use-hardtriplet', dest='use_hardtriplet', action='store_true', help='use triplet loss with hard mining')
    parser.add_argument('--use-amp', dest='use_amp', action='store_true', help='use Automatic Mixed Precision')
    parser.add_argument('--out_dir', type=str, default='output', help='output directory (do not save if out_dir is empty)')
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() and not args.not_use_gpu else torch.device("cpu")

    logger = set_logger("main", args.out_dir)
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))
        
    train(args)

'''bash
CUDA_VISIBLE_DEVICES=1 python trainer.py --use-amp
'''