#!/usr/bin/python3

import torch
from torch import nn, optim

from apex import amp

import sys
import os
## specify gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from tqdm import tqdm
import time
import datetime
import argparse
import numpy as np
import logging
from utils.log import set_logger
from utils.metric_logger import MetricLogger
from utils.checkpoint import Checkpointer
from utils.distribute import synchronize, get_rank, reduce_loss_dict
from solver import make_optimizer
from solver import make_scheduler

import net.network as ntk
from net import pool_aggregator as pa
from data.fashion import FashionDataset
from torch.utils.data import DataLoader
from data.dataloader import get_loader
from data.sampler import make_dataloader

nets = {
    "alexnet": ntk.AlexNet,
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def train(args):
    logger = logging.getLogger("main.trainer")
    try:
        model = nets[args.net](args.margin, args.omega, args.use_hardtriplet)
        model.to(args.device)
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return
    logger.info("Training {}.".format(args.net))

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    if args.device != torch.device("cpu"):
        amp_opt_level = 'O1' if args.use_amp else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], output_device=args.local_rank,
            ## this should be removed if we broadcast BatchNorm stats
            broadcast_buffers=False,
        )
        model = model.module

    arguments = {}
    arguments.update(vars(args))
    arguments["itr"] = 0
    checkpointer = Checkpointer(model, 
                                optimizer=optimizer, 
                                scheduler=scheduler,
                                save_dir=args.out_dir, 
                                save_to_disk=(get_rank()==0))
    ## load model from pretrained_weights or training break_point.
    extra_checkpoint_data = checkpointer.load(args.pretrained_weights)
    arguments.update(extra_checkpoint_data)
    
    batch_size = args.batch_size
    fashion = FashionDataset()
    dataloader = make_dataloader(
        dataset=fashion, 
        images_per_gpu=batch_size,
        num_iters=args.iteration_num,
        start_iter=arguments["itr"],
        shuffle=True,
        is_distributed=args.distributed
    )

    model.train()
    meters = MetricLogger(delimiter=", ")
    max_itr = args.iteration_num
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

        ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        meters.update(**loss_dict_reduced)

        arguments["itr"] = itr
        meters.update(**loss_dict)
        itr_time = time.time() - itr_start_time
        itr_start_time = time.time()
        meters.update(itr_time=itr_time)
        if itr % 50 == 0:
            eta_seconds = meters.itr_time.avg * (max_itr - itr)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join(
                    [
                        "itr: {itr}/{max_itr}",
                        "lr: {lr:.7f}",
                        "{meters}",
                        "eta: {eta}\n",
                        "max mem.: {memory:.0f} MB",
                    ]
                ).format(
                    itr=itr,
                    lr=optimizer.param_groups[0]["lr"],
                    max_itr=max_itr,
                    meters=str(meters),
                    eta=eta,
                    memory=torch.cuda.max_memory_allocated()/1024.0/1024.0,
                )
            )

        ## save model
        if itr % args.checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(itr), **arguments)
        if itr == max_itr:
            checkpointer.save("model_final", **arguments)
            break

    training_time = time.time() - training_start_time
    training_time = str(datetime.timedelta(seconds=int(training_time)))
    logger.info("total training time: {}".format(training_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
    parser.add_argument('--net', type=str, default="vggnet", help='select alexnet/vggnet/resnet')
    parser.add_argument('--not_use_gpu', dest='not_use_gpu', action='store_true', help='do not use gpu')
    parser.add_argument('--iteration_num', type=int, default=250000, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--checkpoint_period', type=int, default=10000, help='period to save checkpoint')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[150000, 220000], help='milestones for lr_scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for lr_scheduler')
    parser.add_argument('--margin', type=float, default=0.2, help='margin')
    parser.add_argument('--omega', type=float, default=0.7, help='weight between losses')
    parser.add_argument('--use-hardtriplet', dest='use_hardtriplet', action='store_true', help='use triplet loss with hard mining')
    parser.add_argument('--use-amp', dest='use_amp', action='store_true', help='use Automatic Mixed Precision')
    parser.add_argument('--pretrained_weights', default=None, help='path to pretrained_weights.pth')
    parser.add_argument('--out_dir', type=str, default='', help='output directory (do not save if out_dir is empty)')
    parser.add_argument("--local_rank", type=int, default=0, help="parameter for torch.distributed.launch")
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() and not args.not_use_gpu else torch.device("cpu")

    ## distribute
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger = set_logger("main", args.out_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))
        
    train(args)

'''bash
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS trainer.py  --net=resnet --out_dir=output --use-amp --batch_size=2
unset NGPUS
'''