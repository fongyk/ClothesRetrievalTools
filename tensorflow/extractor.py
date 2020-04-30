#!/usr/bin/python3

import os
## disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
## specify gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import tensorflow as tf
from tensorflow import keras

strategy = tf.distribute.MirroredStrategy()

import sys
import time
import datetime
import argparse
import json
import numpy as np
from base.utils.log import set_logger
from base.utils.pca import PCAW
from base.data.test_data import load_test_data
from base.config.defaults import INPUT_SIZE
import base.net.network as ntk

parser = argparse.ArgumentParser(description='Extract Deep Feature')
parser.add_argument('--net', type=str, default="vggnet", help='select vggnet/resnet')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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

nets = {
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def write_result(batch_features, batch_predicts, batch_names):
    if isinstance(batch_features, tf.Tensor):
        batch_features = batch_features.numpy()
    if isinstance(batch_names, tf.Tensor):
        batch_names = batch_names.numpy()
    if isinstance(batch_predicts, tf.Tensor):
        batch_predicts = batch_predicts.numpy()
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

logger = set_logger("extractor", args.out_dir)
logger.info("Using {} GPUs".format(strategy.num_replicas_in_sync))
for name, val in vars(args).items():
    logger.info("[PARAMS] {}: {}".format(name, val))

logger.info("Loading test_data ......")
global_batch_size = args.batch_size * strategy.num_replicas_in_sync
query_data, query_len = load_test_data(args.query_data, global_batch_size)
gallery_data, gallery_len = load_test_data(args.gallery_data, global_batch_size)
query_data = strategy.experimental_distribute_dataset(query_data)
gallery_data = strategy.experimental_distribute_dataset(gallery_data)

keras.backend.set_learning_phase(False)
with strategy.scope():
    try:
        model = nets[args.net]()
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        sys.exit(0)
    logger.info("Extracting {} feature.".format(args.net))
    ## load weights
    try:
        latest = tf.train.latest_checkpoint(args.out_dir)
        logger.info("Loading pretrained weights from {}".format(latest))
        model.load_weights(latest).expect_partial()
    except Exception as e:
        logger.info(e)
        logger.info("Loading failed, using initialized weights")
        model.trainable = False
    
    def extract(batch_imgs, batch_names):
        batch_features, batch_predicts = model(batch_imgs, training=False)
        write_result(batch_features, batch_predicts, batch_names)
    
    # @tf.function
    def distributed_extract(*args):
        return strategy.experimental_run_v2(extract, args=(*args,))

logger.info("Start extracting ......")

for test_data, len_data in [(query_data, query_len), (gallery_data, gallery_len)]:
    progbar = keras.utils.Progbar(len_data)
    for i, batch_data in enumerate(test_data, 1):
        progbar.update(i)
        distributed_extract(*batch_data)
        ## To Remove. The break deals with RuntimeError
        if i >= len_data: break

'''load_weights
`trainer` matches `extracter`.
'''

'''
python extractor.py --net=resnet -out_dir=output
'''