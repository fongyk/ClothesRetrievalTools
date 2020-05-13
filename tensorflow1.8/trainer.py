#!/usr/bin/python3

import os
## disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
import time
import datetime
import argparse
import json
import numpy as np
from base.utils.log import set_logger
from base.utils.metric_logger import MetricLogger
from base.data.train_data import load_train_triplet, load_train_pair_with_two_targets
from base.config.defaults import INPUT_SIZE, HEAD_CLS
from base.net.network import Model
from base.net.loss import triplet_loss
from base.solver import make_optimizer

parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
parser.add_argument('--ckpt', type=str, default='resnet_v1_50.ckpt', help='pretrained checkpointer (.ckpt)')
parser.add_argument('--train_num', type=int, default=100000, help='num of training samples to load')
parser.add_argument('--val_num', type=int, default=3000, help='size of validation set')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
parser.add_argument('--out_dir', type=str, default='output', help='output directory')
parser.add_argument('--use-cpu', dest='use_cpu', action='store_true', help='this is tf-gpu, use-cpu to run on cpu')
args, _ = parser.parse_known_args()
if args.use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = set_logger("trainer", args.out_dir)
for name, val in vars(args).items():
    logger.info("[PARAMS] {}: {}".format(name, val))

def compute_accuracy(y_pred, y_true):
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

tf.reset_default_graph()
try:
    xentropy_loss = tf.losses.sparse_softmax_cross_entropy
    ## model forward
    model = Model.model
    input_shape = [None, INPUT_SIZE, INPUT_SIZE, 3]
    anchor = tf.placeholder(tf.float32, input_shape)
    pos = tf.placeholder(tf.float32, input_shape)
    neg = tf.placeholder(tf.float32, input_shape)
    label = tf.placeholder(tf.int32, [None,])
    anchor_feature, anchor_logits = model(anchor, is_training=True, reuse=None)
    pos_feature, pos_logits = model(pos, is_training=True, reuse=True)
    neg_feature, neg_logits = model(neg, is_training=True, reuse=True)
    ## loss
    feature_loss = triplet_loss(
        anchor_feature,
        pos_feature,
        neg_feature,
        args.margin
    )
    logits_loss = 0.5 * (xentropy_loss(label, anchor_logits) + xentropy_loss(label, neg_logits))
    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
    total_loss = args.omega * feature_loss + (1 - args.omega) * logits_loss + args.weight_decay * regularization_loss
    ## accuracy
    anchor_accuracy = compute_accuracy(anchor_logits, label)
    pos_accuracy = compute_accuracy(pos_logits, label)
    neg_accuracy = compute_accuracy(neg_logits, label)
    total_accuracy = (anchor_accuracy + pos_accuracy + neg_accuracy) / 3.0
    ## optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer, lr_schedule = make_optimizer(args, global_step)
    add_global_step = global_step.assign_add(1)
    ## update moving_mean and moving_variance in batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss)
    ## model summary
    # Model.model_summary()
except Exception as e:
    logger.error("Initializing error: {}".format(e))
    sys.exit(0)

arguments = {}
arguments.update(vars(args))
arguments["itr"] = 0
logger.info("Loading train_data ......")
train_data = load_train_triplet(args.batch_size, args.train_num)
## split train and validation set
val_data = train_data.take(args.val_num).repeat()
train_data = train_data.skip(args.val_num)
train_iterator = train_data.make_one_shot_iterator()
next_train_data = train_iterator.get_next()
val_iterator = val_data.make_one_shot_iterator()
next_val_data = val_iterator.get_next()
## train model
logger.info("Start training ......")
meters = MetricLogger(delimiter=", ")
max_itr = args.train_num
start_itr = arguments["itr"] + 1
itr_start_time = time.time()
training_start_time = time.time()

init = tf.global_variables_initializer()
init_scope = [HEAD_CLS, "global_step"]
restorer = Model.make_variable_restorer(exclude_scope=init_scope)
saver = tf.train.Saver(max_to_keep=3)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ## initialize weights
    sess.run(init)
    restorer.restore(sess, args.ckpt)
    ## train
    tf.train.start_queue_runners(sess=sess)
    for itr in range(start_itr, max_itr+1):
        batch_data = sess.run(next_train_data)
        _, batch_tl, batch_xl, batch_accu, add, lr = sess.run(
            fetches=[
                train_op, 
                feature_loss, logits_loss, 
                total_accuracy, 
                add_global_step, lr_schedule
            ],
            feed_dict={
                anchor: batch_data[0],
                pos: batch_data[1],
                neg: batch_data[2],
                label: batch_data[3]
            }
        )
        train_dict={
            "total_loss": args.omega * batch_tl + (1 - args.omega) * batch_xl,
            "feature_loss": batch_tl,
            "logits_loss": batch_xl,
            "train_accuracy": batch_accu,
        }
        arguments["itr"] = itr
        meters.update(**train_dict)
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
                    ]
                ).format(
                    itr=itr,
                    lr=lr,
                    max_itr=max_itr,
                    meters=str(meters),
                    eta=eta
                )
            )

        ## save model
        if itr % args.checkpoint_period == 0:
            saver.save(
                sess, 
                "{}/model.ckpt".format(args.out_dir),
                global_step=itr
            )
            ## validation
            meters.reset("val_accuracy")
            for _ in range(args.val_num):
                batch_data = sess.run(next_val_data)
                val_accu = sess.run(
                    fetches=total_accuracy,
                    feed_dict={
                        anchor: batch_data[0],
                        pos: batch_data[1],
                        neg: batch_data[2],
                        label: batch_data[3]
                    }
                )
                meters.update(val_accuracy=val_accu)
            logger.info("val_accuracy: {:.2f}\n".format(meters.val_accuracy.global_avg))
            
        if itr == max_itr:
            saver.save(
                sess, 
                "{}/model_final.ckpt".format(args.out_dir),
                global_step=itr
            )
            with open(os.path.join(args.out_dir, "arguments.json"), "w") as fw:
                json.dump(arguments, fw)
            break

    training_time = time.time() - training_start_time
    training_time = str(datetime.timedelta(seconds=int(training_time)))
    logger.info("Total training time: {}".format(training_time))


'''GPU
CUDA_VISIBLE_DEVICES=1 python trainer.py  --out_dir=output
'''

'''CPU
CUDA_VISIBLE_DEVICES=-1 python trainer.py --out_dir=output
'''