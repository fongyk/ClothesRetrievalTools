#!/usr/bin/python3

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import net.functional as Fn

from apex import amp

import sys
import os
from tqdm import tqdm
import time
import datetime
import argparse
import numpy as np
from utils.log import logger
from utils.metric_logger import MetricLogger

import net.network as ntk
from net import pool_aggregator as pa
from data.fashion import FashionDataset
from torch.utils.data import DataLoader
from data.dataloader import get_loader

nets = {
    "alexnet": ntk.AlexNet,
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def adjust_learning_rate(optimizer, epoch_th, epoch_num):
    if epoch_th == int(epoch_num * 0.5):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    if epoch_th == int(epoch_num * 0.8):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def train(args):
    try:
        model = nets[args.model](args.margin, args.omega)
        model.to(args.device)
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return
    logger.info("Training {}.".format(args.model))

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.learning_rate},
    ])

    amp_opt_level = 'O1' if args.use_amp else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    
    batch_size = args.batch_size
    fashion = FashionDataset()
    dataloader = DataLoader(dataset=fashion, shuffle=True, num_workers=8, batch_size=batch_size)

    model.train()
    meters = MetricLogger(delimiter=", ")
    max_itr = args.epoch_num * len(dataloader)
    itr = 0
    itr_start_time = time.time()
    training_start_time = time.time()
    for e in range(1, 1+args.epoch_num):
        for batch_id, batch_data in enumerate(dataloader):

            batch_data = (bd.to(args.device) for bd in batch_data)
            loss_dict = model.loss(*batch_data)
            optimizer.zero_grad()
            # loss_dict["loss"].backward()
            with amp.scale_loss(loss_dict["loss"], optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()

            itr += 1
            meters.update(**loss_dict)
            itr_time = time.time() - itr_start_time
            itr_start_time = time.time()
            meters.update(itr_time=itr_time)
            if itr % 50 == 0:
                eta_seconds = meters.itr_time.global_avg * (max_itr - itr)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    meters.delimiter.join(
                        [
                            "epoch: {e}/{epoch_num}",
                            "itr: {itr}/{max_itr}",
                            "lr: {lr:.7f}",
                            "{meters}",
                            "eta: {eta}\n",
                        ]
                    ).format(
                        e=e,
                        epoch_num=args.epoch_num,
                        itr=itr,
                        lr=optimizer.param_groups[0]["lr"],
                        max_itr=max_itr,
                        meters=str(meters),
                        eta=eta,
                    )
                )

        adjust_learning_rate(optimizer, e, args.epoch_num)
        
        ## save model
        torch.save({"model_state_dict": model.state_dict(),
                    }, 
                    "output/model_{:02d}.pth".format(e))

    training_time = time.time() - training_start_time
    training_time = str(datetime.timedelta(seconds=int(training_time)))
    logger.info("total training time: {}".format(training_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
    parser.add_argument('--model', type=str, required=False, default="vggnet", help='select AlexNet/VGGNet/ResNet')
    parser.add_argument('--epoch_num', type=int, default=15, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning_rate')
    parser.add_argument('--margin', type=float, default=0.1, help='margin')
    parser.add_argument('--omega', type=float, default=0.5, help='weight between losses')
    parser.add_argument('--use-amp', dest='use_amp', action='store_true', help='use Automatic Mixed Precision')
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for par, val in args.__dict__.items():
        logger.info("[PARAMS] {}: {}".format(par, val))
        
    train(args)
