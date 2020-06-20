#!/usr/bin/python3

import os
## disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
## specify GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
GPUs = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
num_gpu = len(GPUs)

import tensorflow as tf

import sys
import time
import datetime
import argparse
import json
import numpy as np
from tqdm import tqdm
from base.utils.log import set_logger
from base.utils.pca import PCAW
from base.data.test_data import load_test_data
from base.config.defaults import INPUT_SIZE
from base.net.network import Model

parser = argparse.ArgumentParser(description='Extract Deep Feature')
parser.add_argument('--batch_size', type=int, default=32, help='batch size per gpu')
parser.add_argument('--query_data', type=str, default='fashion_crop_validation_query_list.txt', help='path to test dataset-list')
parser.add_argument('--gallery_data', type=str, default='fashion_crop_validation_gallery_list.txt', help='path to test dataset-list')
parser.add_argument('--pcaw_path', default=None, help='path to pca-whiten params: mean.npy & pcaw.npy')
parser.add_argument('--pcaw_dims', type=int, default=None, help='number of principal components')
parser.add_argument('--out_dir', type=str, default='output', help='pretrained models directory as well as feature output directory')
args, _ = parser.parse_known_args()

if args.pcaw_path:
    m = np.load(os.path.join(args.pcaw_path, "mean.npy"))
    P = np.load(os.path.join(args.pcaw_path, "pcaw.npy"))
    pcaw = PCAW(m, P, args.pcaw_dims)
    args.pcaw = pcaw
else:
    args.pcaw = None

logger = set_logger("extractor", args.out_dir)
logger.info("Using {} GPUs".format(num_gpu))
for name, val in vars(args).items():
    logger.info("[PARAMS] {}: {}".format(name, val))

def write_result(batch_features, batch_predicts, batch_names):
    if args.pcaw is not None:
        batch_features = args.pcaw(batch_features.T, transpose=True)
    batch_predicts = np.argmax(batch_predicts, axis=1)
    for feature_per_image, predict_per_image, name_per_image in zip(batch_features, batch_predicts, batch_names):
        ## convert name from `bytes` to `str`
        name_per_image = name_per_image.decode("utf-8")
        try:
            out_dir = os.path.join(args.out_dir, 'feat')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            np.save(os.path.join(out_dir, name_per_image + '.npy'), feature_per_image)
            np.save(os.path.join(out_dir, name_per_image + '.prd.npy'), predict_per_image)
        except OSError:
            logger.info("Can not write feature with {}.".format(name_per_image))

tf.reset_default_graph()
try:
    ## inference
    model = Model.model
    input_shape = [None, INPUT_SIZE, INPUT_SIZE, 3]
    images = tf.placeholder(tf.float32, input_shape)
    image_splits = tf.split(images, num_or_size_splits=num_gpu, axis=0)
    tower_feature, tower_logits = [], []
    for i, gpu in enumerate(GPUs):
        with tf.device("/gpu:%d" % gpu):
            ## reuse_vars = True
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                feature, logits, large_centers = model(image_splits[i], is_training=False)
                tower_feature.append(feature)
                tower_logits.append(logits)
    tower_feature = tf.concat(tower_feature, axis=0)
    tower_logits = tf.concat(tower_logits, axis=0)

except Exception as e:
    logger.error("Initializing error: {}".format(e))
    sys.exit(0)

saver = tf.train.Saver()
## load weights
latest = tf.train.latest_checkpoint(args.out_dir)
if latest is not None:
    logger.info("Loading pretrained weights from {}".format(latest))
else:
    logger.info("Loading failed, using initialized weights")
    init = tf.global_variables_initializer()

logger.info("Loading test_data ......")
global_batch_size = args.batch_size * num_gpu
query_data, query_len = load_test_data(args.query_data, global_batch_size)
gallery_data, gallery_len = load_test_data(args.gallery_data, global_batch_size)
query_iterator = query_data.make_one_shot_iterator()
gallery_iterator = gallery_data.make_one_shot_iterator()
next_query = query_iterator.get_next()
next_gallery = gallery_iterator.get_next()
logger.info("Start extracting ......")
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    if latest is not None:
        saver.restore(sess, latest)
    else:
        sess.run(init)
    for test_data, test_len in ((next_query, query_len), (next_gallery, gallery_len)):
        for i in tqdm(range(test_len)):
            centers = large_centers.eval()
            logger.debug(centers[:5, :5])
            batch_imgs, batch_names = sess.run(test_data)
            batch_features, batch_predicts = sess.run(
                fetches=[tower_feature, tower_logits],
                feed_dict={
                    images: batch_imgs
                }
            )
            write_result(
                batch_features,
                batch_predicts,
                batch_names
            )
'''
python extractor.py -out_dir=output
'''
