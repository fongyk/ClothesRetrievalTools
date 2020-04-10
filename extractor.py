#!/usr/bin/python3

import torch
import torch.nn as nn
import net.functional as Fn

import sys
import os
from tqdm import tqdm
import time
import argparse
import numpy as np
from utils.log import logger

import net.network as ntk
from data.dataloader import get_loader

nets = {
    "alexnet": ntk.AlexNet,
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def extract_image_feature(args):
    try:
        model = nets[args.model]()
        model.to(args.device)
    except Exception as e:
        logger.error(e, "== Not Implemented ==")
        return 
    logger.info("Extracting {} feature.".format(args.model))

    query_dataloader = get_loader(args.query_data, args.batch_size)
    gallery_dataloader = get_loader(args.gallery_data, args.batch_size)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        for dataloader in [query_dataloader, gallery_dataloader]:
            for batch_imgs, batch_filenames in tqdm(dataloader):
                batch_imgs = batch_imgs.to(args.device)
                batch_features, batch_predicts = model(batch_imgs)
                batch_features = batch_features.cpu().detach().numpy()
                batch_predicts = np.argmax(batch_predicts.cpu().detach().numpy(), axis=1)
                for feature_per_image, predict_per_image, name_per_image in zip(batch_features, batch_predicts, batch_filenames):
                    try:
                        np.save(os.path.join('feat', name_per_image + '.npy'), feature_per_image)
                        np.save(os.path.join('feat', name_per_image + '.prd.npy'), predict_per_image)
                    except OSError:
                        logger.info("can not write feature with {}.".format(name_per_image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Deep Feature')
    parser.add_argument('--model', type=str, required=False, default="vggnet", help='select AlexNet/VGGNet/ResNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--query_data', type=str, default='fashion_crop_validation_query_list.txt', help='path to test dataset-folder or dataset-list')
    parser.add_argument('--gallery_data', type=str, default='fashion_crop_validation_gallery_list.txt', help='path to test dataset-folder or dataset-list')
    parser.add_argument('--checkpoint_path', type=str, default='output/models_01.pth', help='checkpoint path for extracting feature.')
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for par, val in args.__dict__.items():
        logger.info("[PARAMS] {}: {}".format(par, val))

    extract_image_feature(args)

'''bash
CUDA_VISIBLE_DEVICES=1 python extractor.py -d img.list --model vggnet
'''