#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from tensorflow import keras

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

nets = {
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def write_result(logger, args, batch_features, batch_predicts, batch_names):
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

def extract(args):
    logger = set_logger("extractor", args.out_dir)
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))
    try:
        model = nets[args.net]()
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return 
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
    keras.backend.set_learning_phase(False)

    logger.info("Loading test_data ......")
    query_data = load_test_data(args.query_data, args.batch_size)
    gallery_data = load_test_data(args.gallery_data, args.batch_size)
    logger.info("Start extracting ......")
    for test_data in [query_data, gallery_data]:
        len_data = int(tf.data.experimental.cardinality(test_data))
        progbar = keras.utils.Progbar(len_data)
        for i, (batch_imgs, batch_names) in enumerate(test_data, 1):
            progbar.update(i)
            # batch_imgs = tf.stop_gradient(batch_imgs)
            batch_features, batch_predicts = model(batch_imgs, training=False)
            write_result(
                logger, 
                args, 
                batch_features, 
                batch_predicts, 
                batch_names
            )

def predict(args):
    logger = set_logger("extractor", args.out_dir)
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))
    try:
        base_model = nets[args.net]()
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return 
    logger.info("Extracting {} feature.".format(args.net))

    ## construct model
    input_shape = (INPUT_SIZE, INPUT_SIZE, 3)
    inputs = keras.layers.Input(shape=input_shape)
    feature, logits = base_model.embedder(inputs)
    model = keras.Model(
        inputs=inputs,
        outputs=[feature, logits],
        name="retrieval"
        )
    model.summary()
    keras.utils.plot_model(model, '{}/model.png'.format(args.out_dir), show_shapes=True)
    
    ## load weights
    try:
        latest = tf.train.latest_checkpoint(args.out_dir)
        logger.info("Loading pretrained weights from {}".format(latest))
        model.load_weights(latest).expect_partial()
        # model = keras.models.load_model(os.path.join(args.out_dir, "model_final.h5"), compile=False)
    except Exception as e:
        logger.info(e)
        logger.info("Loading failed, using initialized weights")
    keras.backend.set_learning_phase(False)

    logger.info("Loading test_data ......")
    query_data = load_test_data(args.query_data, args.batch_size)
    gallery_data = load_test_data(args.gallery_data, args.batch_size)
    logger.info("Start extracting ......")
    for test_data in [query_data, gallery_data]:
        len_data = int(tf.data.experimental.cardinality(test_data))
        progbar = keras.utils.Progbar(len_data)
        for i, (batch_imgs, batch_names) in enumerate(test_data, 1):
            progbar.update(i)
            batch_features, batch_predicts = model.predict(batch_imgs)
            write_result(
                logger, 
                args, 
                batch_features, 
                batch_predicts, 
                batch_names
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Deep Feature')
    parser.add_argument('--net', type=str, default="vggnet", help='select vggnet/resnet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--query_data', type=str, default='fashion_crop_validation_query_list.txt', help='path to test dataset-list')
    parser.add_argument('--gallery_data', type=str, default='fashion_crop_validation_gallery_list.txt', help='path to test dataset-list')
    parser.add_argument('--pcaw_path', default=None, help='path to pca-whiten params: mean.npy & pcaw.npy')
    parser.add_argument('--pcaw_dims', type=int, default=None, help='number of principal components')
    parser.add_argument('--out_dir', type=str, default='output', help='pretrained models directory as well as feature output directory')
    parser.add_argument('--use-predict', dest='use_predict', action='store_true', help='use `predict` mode (default: `extract` mode)')
    parser.add_argument('--use-cpu', dest='use_cpu', action='store_true', help='this is tf-gpu, use-cpu to run on cpu')
    args, _ = parser.parse_known_args()

    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

    if args.pcaw_path:
        m = np.load(os.path.join(args.pcaw_path, "mean.npy"))
        P = np.load(os.path.join(args.pcaw_path, "pcaw.npy"))
        pcaw = PCAW(m, P, args.pcaw_dims)
        args.pcaw = pcaw
    else:
        args.pcaw = None

    if args.use_predict:
        predict(args)
    else:
        extract(args)

'''load_weights
`train` in trainer matches `extract` in extractor.
`fit` in trainner matches `predict` in extractor.
'''

'''GPU
CUDA_VISIBLE_DEVICES=1 python extractor.py --net=resnet -out_dir=output
'''

'''CPU
CUDA_VISIBLE_DEVICES=-1 python extractor.py --net=resnet -out_dir=output
'''