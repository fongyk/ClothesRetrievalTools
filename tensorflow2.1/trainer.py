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

parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
parser.add_argument('--net', type=str, default="vggnet", help='select vggnet/resnet')
parser.add_argument('--train_num', type=int, default=100000, help='num of training samples to load')
parser.add_argument('--val_num', type=int, default=1000, help='size of validation set')
parser.add_argument('--batch_size', type=int, default=32, help='batch size per gpu')
parser.add_argument('--checkpoint_period', type=int, default=10000, help='period to save checkpoint')
parser.add_argument('--optimizer', type=str, default="sgd", help='optimizer: Adam (lr: 5e-5), SGD (lr: 0.01), ...')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--scheduler', type=str, default="exp", help='lr_scheduler: exp(ExponentialDecay), piece(PiecewiseConstantDecay)')
parser.add_argument('--lr_milestones', type=int, nargs='+', default=[60000, 80000], help='milestones for piece_scheduler')
parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for piece_scheduler')
parser.add_argument('--decay_steps', type=int, default=10000, help='decay_steps for exp_scheduler')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay_rate for exp_scheduler')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--margin', type=float, default=0.2, help='margin')
parser.add_argument('--omega', type=float, default=0.7, help='weight between losses')
parser.add_argument('--out_dir', type=str, default='output', help='output directory (do not save if out_dir is empty)')
args, _ = parser.parse_known_args()

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

logger = set_logger("trainer", args.out_dir)
logger.info("Using {} GPUs".format(strategy.num_replicas_in_sync))
for name, val in vars(args).items():
    logger.info("[PARAMS] {}: {}".format(name, val))
logger.info("Loading train_data ......")
global_batch_size = args.batch_size * strategy.num_replicas_in_sync
train_data = load_train_triplet(global_batch_size, args.train_num)
## split train and validation set
val_data = train_data.take(args.val_num)
train_data = train_data.skip(args.val_num)
train_data = strategy.experimental_distribute_dataset(train_data)
val_data = strategy.experimental_distribute_dataset(val_data)

keras.backend.set_learning_phase(True)
with strategy.scope():
    try:
        model = nets[args.net]()
        siamese_model = ntk.build_threestream_siamese_network(model)
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        sys.exit(0)
    logger.info("Training {}.".format(args.net))
    triplet_loss = net_loss.TripletLoss(args.margin, reduction=tf.keras.losses.Reduction.NONE)
    xentropy_loss = keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = make_optimizer(args)
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

    def val_step(
        anchor, 
        pos, 
        neg, 
        label, 
    ):
        anchor_feature, anchor_logits = model(anchor, training=True)
        neg_feature, neg_logits = model(neg, training=True)
        val_acc.update_state(label, anchor_logits)
        val_acc.update_state(label, neg_logits)

    @tf.function
    def distributed_val_step(*args):
        return strategy.experimental_run_v2(val_step, args=(*args,))

    def train_step(
        omega,
        anchor, 
        pos, 
        neg, 
        label, 
    ):
        def compute_xentropy_loss(y_true, y_pred):
            loss_per_batch = xentropy_loss(y_true, y_pred)
            return tf.nn.compute_average_loss(
                loss_per_batch,
                global_batch_size=global_batch_size
            )
        with tf.GradientTape() as tape:
            anchor_out, pos_out, neg_out = siamese_model([anchor, pos, neg])
            anchor_feature, anchor_logits = anchor_out
            pos_feature, pos_logits = pos_out
            neg_feature, neg_logits = neg_out
            loss_feature = triplet_loss(label, [anchor_feature, pos_feature, neg_feature]) * (1.0 / global_batch_size)
            loss_logits = 0.5 * (compute_xentropy_loss(label, anchor_logits) + compute_xentropy_loss(label, neg_logits))
            loss_total = [omega * loss_feature, (1 - omega) * loss_logits]

        gradients = tape.gradient(loss_total, siamese_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, siamese_model.trainable_variables))

        train_acc.update_state(label, anchor_logits)
        train_acc.update_state(label, neg_logits)

        return sum(loss_total), loss_feature, loss_logits

    @tf.function
    def distributed_train_step(*args):
        losses_per_gpu = strategy.experimental_run_v2(train_step, args=(*args,))
        loss_total, loss_feature, loss_logits = [
            strategy.reduce(
                tf.distribute.ReduceOp.SUM, 
                loss_per_gpu, 
                axis=None
            ) for loss_per_gpu in losses_per_gpu
        ]
        return dict(
            loss_total=loss_total, 
            loss_feature=loss_feature, 
            loss_logits=loss_logits
        )

siamese_model.summary()
keras.utils.plot_model(siamese_model, '{}/siamese_model.png'.format(args.out_dir), show_shapes=True)

arguments = {}
arguments.update(vars(args))
arguments["itr"] = 0
## train model
logger.info("Start training ......")
meters = MetricLogger(delimiter=", ")
max_itr = args.train_num
start_itr = arguments["itr"] + 1
itr_start_time = time.time()
training_start_time = time.time()
for itr, batch_data in enumerate(train_data, start_itr):
    loss_dict = distributed_train_step(
        args.omega, 
        *batch_data
    )
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
                    "train_accuracy: {train_acc:.2f}",
                    "eta: {eta}\n",
                ]
            ).format(
                itr=itr,
                # lr=optimizer.lr.numpy(),
                lr=optimizer._decayed_lr(tf.float32),
                max_itr=max_itr,
                meters=str(meters),
                train_acc=train_acc.result().numpy(),
                eta=eta
            )
        )

    ## save model
    if itr % args.checkpoint_period == 0:
        model.save_weights("{}/model_{:07d}.ckpt".format(args.out_dir, itr))
        ## validation
        for batch_data in val_data:
            distributed_val_step(*batch_data)
        logger.info("val_accuracy: {:.2f}\n".format(val_acc.result().numpy()))
        train_acc.reset_states()
        val_acc.reset_states()

    if itr == max_itr:
        model.save_weights("{}/model_final.ckpt".format(args.out_dir))
        with open(os.path.join(args.out_dir, "arguments.json"), "w") as fw:
            json.dump(arguments, fw)
        break

training_time = time.time() - training_start_time
training_time = str(datetime.timedelta(seconds=int(training_time)))
logger.info("total training time: {}".format(training_time))


'''
python trainer.py --net=resnet -out_dir=output
'''