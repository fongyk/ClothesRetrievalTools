#!/usr/bin/python3

import os
## disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
## specify gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

strategy = tf.distribute.MirroredStrategy()

import sys
import time
import datetime
import argparse
import json
import numpy as np
from base.utils.log import set_logger
from base.utils.metric_logger import MetricLogger
from base.data.train_data import load_train_triplet, load_train_pair, load_train_pair_from_numpy, load_train_pair_with_two_targets
from base.config.defaults import INPUT_SIZE, HEAD_FT, HEAD_CLS
import base.net.network as ntk
from base.net import loss as net_loss
from base.solver import make_optimizer, make_callbacks

nets = {
    "vggnet": ntk.VGGNet,
    "resnet": ntk.ResNet,
}

def get_bn_vars(collection):
    moving_mean, moving_variance = None, None
    for var in collection:
        name = var.name.lower()
        if "variance" in name:
            moving_variance = var
        if "mean" in name:
            moving_mean = var
    if moving_mean is not None and moving_variance is not None:
        return moving_mean, moving_variance
    raise ValueError("Unable to find moving mean and variance")

def fit(args):
    logger = set_logger("fitter", args.out_dir)
    logger.info("Using {} GPUs".format(strategy.num_replicas_in_sync))
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))

    keras.backend.set_learning_phase(True)
    ## construct model
    with strategy.scope():
        try:
            base_model = nets[args.net]()
        except Exception as e:
            logger.error("Initialize {} error: {}".format(args.net, e))
            return
        logger.info("Training {}.".format(args.net))

        losses = {
            HEAD_FT: net_loss.TripletSemiHardLoss(args.margin),
            HEAD_CLS: keras.losses.SparseCategoricalCrossentropy()
        }
        loss_weights = {
            HEAD_FT: args.omega,
            HEAD_CLS: 1 - args.omega
        }
        metrics={HEAD_CLS: [keras.metrics.SparseCategoricalAccuracy()]}
        if args.optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        elif args.optimizer == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate)
        else:
            raise NotImplementedError("Choose Adam or SGD")
        input_shape = (INPUT_SIZE, INPUT_SIZE, 3)
        inputs = keras.layers.Input(shape=input_shape)
        feature, logits = base_model.embedder(inputs)
        model = keras.Model(
            inputs=inputs,
            outputs=[feature, logits],
            name="retrieval"
            )
        ## compile model
        model.compile(
            optimizer=optimizer, 
            loss=losses, 
            loss_weights=loss_weights,
            metrics=metrics
        )

    model.summary()
    keras.utils.plot_model(model, '{}/model.png'.format(args.out_dir), show_shapes=True)

    ## load data
    logger.info("Loading train_data ......")
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    train_data = load_train_pair_with_two_targets(global_batch_size, args.train_num)
    val_data = train_data.take(args.val_num)
    train_data = train_data.skip(args.val_num)

    ## train model
    logger.info("Start training ......")
    callbacks = make_callbacks(args)
    H = model.fit(
        x=train_data.repeat(),
        validation_data=val_data,
        epochs=args.epoch_num,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    train_acc = H.history["logits_sparse_categorical_accuracy"]
    val_acc = H.history["val_logits_sparse_categorical_accuracy"]
    logger.info("train_accuracy: {}".format(list(map(lambda e:round(e,3), train_acc))))
    logger.info("val_accuracy: {}".format(list(map(lambda e:round(e,3), val_acc))))

    model.save(os.path.join(args.out_dir, "model_final.h5"))
    with open(os.path.join(args.out_dir, "arguments.json"), "w") as fw:
        json.dump(vars(args), fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
    parser.add_argument('--net', type=str, default="vggnet", help='select vggnet/resnet')
    parser.add_argument('--train_num', type=int, default=100000, help='num of training samples to load')
    parser.add_argument('--val_num', type=int, default=1000, help='size of validation set')
    parser.add_argument('--epoch_num', type=int, default=10, help='epoch num')
    parser.add_argument('--steps_per_epoch', type=int, default=5000, help='steps_per_epoch for `fit` mode')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
    parser.add_argument('--optimizer', type=str, default="sgd", help='optimizer: Adam (lr: 5e-5), SGD (lr: 0.01), ...')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--scheduler', type=str, default="exp", help='lr_scheduler: exp(ExponentialDecay), piece(PiecewiseConstantDecay)')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for piece_scheduler')
    parser.add_argument('--margin', type=float, default=0.2, help='margin')
    parser.add_argument('--omega', type=float, default=0.7, help='weight between losses')
    parser.add_argument('--out_dir', type=str, default='output', help='output directory (do not save if out_dir is empty)')
    args, _ = parser.parse_known_args()

    fit(args)

'''
python fitter.py --net=resnet -out_dir=output
'''