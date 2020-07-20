#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
from tqdm import tqdm
import time
import argparse
import numpy as np
from utils.log import set_logger
from utils.checkpoint import Checkpointer
from utils.pca import PCAW

import net.network as ntk
from data.dataloader import get_loader

logger = set_logger("extractor")

nets = {
    "alexnet": ntk.AlexNet,
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def extract_image_feature(args):
    try:
        model = nets[args.net]()
        model.to(args.device)
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return 
    logger.info("Extracting {} feature.".format(args.net))

    query_dataloader = get_loader(args.query_data, args.batch_size)
    gallery_dataloader = get_loader(args.gallery_data, args.batch_size)

    checkpointer = Checkpointer(model, save_dir=args.out_dir)
    _ = checkpointer.load(args.checkpoint_path, use_latest=args.checkpoint_path is None)

    model.eval()
    with torch.no_grad():
        for dataloader in [query_dataloader, gallery_dataloader]:
            for batch_imgs, batch_filenames in tqdm(dataloader):
                batch_imgs = batch_imgs.to(args.device)
                batch_features, batch_predicts = model(batch_imgs)
                batch_features = batch_features.cpu().detach().numpy()
                if args.pcaw is not None:
                    batch_features = args.pcaw(batch_features.T, transpose=True)
                batch_predicts = np.argmax(batch_predicts.cpu().detach().numpy(), axis=1)
                for feature_per_image, predict_per_image, name_per_image in zip(batch_features, batch_predicts, batch_filenames):
                    try:
                        out_dir = os.path.join(args.out_dir, 'feat')
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        np.save(os.path.join(out_dir, name_per_image + '.npy'), feature_per_image)
                        np.save(os.path.join(out_dir, name_per_image + '.prd.npy'), predict_per_image)
                    except OSError:
                        logger.info("can not write feature with {}.".format(name_per_image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Deep Feature')
    parser.add_argument('--net', type=str, default="vggnet", help='select alexnet/vggnet/resnet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--query_data', type=str, default='fashion_crop_validation_query_list.txt', help='path to test dataset-folder or dataset-list')
    parser.add_argument('--gallery_data', type=str, default='fashion_crop_validation_gallery_list.txt', help='path to test dataset-folder or dataset-list')
    parser.add_argument('--checkpoint_path', default=None, help='checkpoint path for extracting feature')
    parser.add_argument('--pcaw_path', default=None, help='path to pca-whiten params: mean.npy & pcaw.npy')
    parser.add_argument('--pcaw_dims', type=int, default=None, help='number of principal components')
    parser.add_argument('--out_dir', type=str, required=True, help='pretrained models directory as well as feature output directory')
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.pcaw_path:
        m = np.load(os.path.join(args.pcaw_path, "mean.npy"))
        P = np.load(os.path.join(args.pcaw_path, "pcaw.npy"))
        pcaw = PCAW(m, P, args.pcaw_dims)
        args.pcaw = pcaw
    else:
        args.pcaw = None

    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))

    extract_image_feature(args)

'''bash
CUDA_VISIBLE_DEVICES=1 python extractor.py  --net=vggnet --checkpoint_path=output/model.pth
'''