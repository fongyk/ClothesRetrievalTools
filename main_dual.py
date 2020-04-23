#!/usr/bin/python3

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import net.functional as Fn

import sys
import os
from tqdm import tqdm
import time
import datetime
import argparse
import numpy as np
from utils.log import logger

import net.network as ntk
from net import pool_aggregator as pa
from data.fashion import FashionDataset, tripletLoss
from torch.utils.data import DataLoader
from data.dataloader import get_loader

nets = {
    "alexnet": ntk.AlexNet,
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

aggregators = {
    "mac": pa.MAC,
    "rmac": pa.RMAC,
    "spoc": pa.SPoC,
    "crow": pa.CroW,
    "gem": pa.GeM,
}


def adjust_learning_rate(optimizer, epoch_th, epoch_num):
    if epoch_th == int(epoch_num * 0.6):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    if epoch_th == int(epoch_num * 0.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def get_feature(image, model, aggre, device):
    image = image.to(device)
    feature_map = model(image)
    feature = aggre(feature_map)
    return feature

def train(args):
    try:
        model_shop = nets[args.model]()
        aggre_shop = aggregators[args.aggre]()
        model_shop.to(args.device)
        aggre_shop.to(args.device)
        model_user = nets[args.model]()
        aggre_user = aggregators[args.aggre]()
        model_user.to(args.device)
        aggre_user.to(args.device)
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.model, e))
        sys.exit(1)
    logger.info("Training {}-{}.".format(args.model, args.aggre))

    
    batch_size = args.batch_size
    fashion = FashionDataset()
    dataloader = DataLoader(dataset=fashion, shuffle=True, num_workers=8, batch_size=batch_size)

    criterion = tripletLoss(args.margin)
    optimizer = optim.Adam([
        {'params': model_shop.parameters(), 'lr': args.learning_rate},
        {'params': aggre_shop.parameters(), 'lr': args.learning_rate},
        {'params': model_user.parameters(), 'lr': args.learning_rate},
        {'params': aggre_user.parameters(), 'lr': args.learning_rate},
    ])

    model_shop.train()
    aggre_shop.train()
    model_user.train()
    aggre_user.train()
    train_loss = 0.0
    max_itr = args.epoch_num * len(dataloader)
    itr = 0
    batch_start_time = time.time()
    training_start_time = time.time()
    for e in range(1, 1+args.epoch_num):
        for batch, (shop_a, user_a, shop_n, user_n) in enumerate(dataloader):
            shop_a_feature = get_feature(shop_a, model_shop, aggre_shop, args.device)
            user_a_feature = get_feature(user_a, model_user, aggre_user, args.device)
            shop_n_feature = get_feature(shop_n, model_shop, aggre_shop, args.device)
            user_n_feature = get_feature(user_n, model_user, aggre_user, args.device)
            loss = criterion(shop_a_feature, user_a_feature, shop_n_feature) + criterion(user_a_feature, shop_a_feature, user_n_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            itr += 1
            train_loss += loss.item()
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            eta_seconds = batch_time * (max_itr - itr)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            if itr % args.check_step == 0:
                logger.info("epoch: {}/{}, itr:{}/{}, loss:{:.4f}, eta:{}".format(e, args.epoch_num, itr, max_itr, train_loss/args.check_step, eta))
                train_loss = 0.0

        adjust_learning_rate(optimizer, e, args.epoch_num)
        
        ## save model
        torch.save({"model_shop_state_dict": model_shop.state_dict(),
                    "aggre_shop_state_dict": aggre_shop.state_dict(),
                    "model_user_state_dict": model_user.state_dict(),
                    "aggre_user_state_dict": aggre_user.state_dict(),
                    }, 
                    "output/models_{:02d}.pth".format(e))

    training_time = time.time() - training_start_time
    training_time = str(datetime.timedelta(seconds=int(training_time)))
    logger.info("total training time: {}".format(training_time))

def extract_feature(args):
    try:
        model_shop = nets[args.model]()
        aggre_shop = aggregators[args.aggre]()
        model_shop.to(args.device)
        aggre_shop.to(args.device)
        model_user = nets[args.model]()
        aggre_user = aggregators[args.aggre]()
        model_user.to(args.device)
        aggre_user.to(args.device)
    except Exception as e:
        logger.error(e, "== Not Implemented ==")
        sys.exit(1)
    logger.info("Extracting {}-{} feature.".format(args.model, args.aggre))

    query_dataloader = get_loader(args.query_data, args.batch_size)
    gallery_dataloader = get_loader(args.gallery_data, args.batch_size)

    checkpoint = torch.load(args.checkpoint_path)
    model_shop.load_state_dict(checkpoint["model_shop_state_dict"])
    aggre_shop.load_state_dict(checkpoint["aggre_shop_state_dict"])
    model_user.load_state_dict(checkpoint["model_user_state_dict"])
    aggre_user.load_state_dict(checkpoint["aggre_user_state_dict"])
    model_shop.eval()
    aggre_shop.eval()
    model_user.eval()
    aggre_user.eval()
    with torch.no_grad():
        for dataloader, model, aggre in [(query_dataloader, model_user, aggre_user), (gallery_dataloader, model_shop, aggre_shop)]:
            for img, filename in tqdm(dataloader):
                img = img.to(args.device)
                feature_map = model(img)
                feature = aggre(feature_map)
                for feat, name in zip(feature, filename):
                    try:
                        np.save(os.path.join('feat', name + '.npy'), feat.cpu().detach().numpy())
                    except OSError:
                        logger.info("can not write feature with {}.".format(name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
    parser.add_argument('--model', type=str, required=False, default="vggnet", help='select AlexNet/VGGNet/ResNet')
    parser.add_argument('--aggre', type=str, default="mac", help='aggregator')
    parser.add_argument('--epoch_num', type=int, default=12, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning_rate')
    parser.add_argument('--margin', type=float, default=0.1, help='margin')
    parser.add_argument('--check_step', type=int, default=50, help='train loss check step')
    parser.add_argument('--query_data', type=str, default='fashion_crop_validation_query_list.txt', help='path to test dataset-folder or dataset-list')
    parser.add_argument('--gallery_data', type=str, default='fashion_crop_validation_gallery_list.txt', help='path to test dataset-folder or dataset-list')
    parser.add_argument('--checkpoint_path', type=str, default='output/models_01.pth', help='checkpoint path for extracting feature.')
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for par, val in args.__dict__.items():
        logger.info("[PARAMS] {}: {}".format(par, val))
    train(args)
    extract_feature(args)

