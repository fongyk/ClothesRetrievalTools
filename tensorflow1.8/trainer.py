#!/usr/bin/python3

import os
## disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
## specify GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
GPUs = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
num_gpu = len(GPUs)

import tensorflow as tf
from tensorflow import keras

import sys
import time
import datetime
import argparse
import json
import numpy as np
from base.utils.log import set_logger
from base.utils.metric_logger import MetricLogger
from base.data.train_data import load_train_triplet, load_train_pair_with_two_targets
from base.config.defaults import INPUT_SIZE, NUM_CLASSES, HEAD_FT, HEAD_CLS
from base.net.network import Model
from base.net.loss import triplet_loss
from base.solver import make_optimizer

parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
parser.add_argument('--ckpt', type=str, default='resnet_v1_50.ckpt', help='pretrained checkpointer (.ckpt)')
parser.add_argument('--train_num', type=int, default=50000, help='num of training samples to load')
parser.add_argument('--val_num', type=int, default=1000, help='size of validation set')
parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
parser.add_argument('--checkpoint_period', type=int, default=5000, help='period to save checkpoint')
parser.add_argument('--optimizer', type=str, default="adam", help='optimizer: Adam (lr: 5e-5), SGD (lr: 0.01), ...')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
parser.add_argument('--scheduler', type=str, default="exp", help='lr_scheduler: exp(ExponentialDecay), piece(PiecewiseConstantDecay)')
parser.add_argument('--lr_milestones', type=int, nargs='+', default=[30000, 42000], help='milestones for piece_scheduler')
parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for piece_scheduler')
parser.add_argument('--decay_steps', type=int, default=10000, help='decay_steps for exp_scheduler')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay_rate for exp_scheduler')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--margin', type=float, default=0.2, help='margin')
parser.add_argument('--omega', type=float, default=0.7, help='weight between losses')
parser.add_argument('--out_dir', type=str, default='output', help='output directory')
args, _ = parser.parse_known_args()
logger = set_logger("distributed_trainer", args.out_dir)
logger.info("Using {} GPUs".format(num_gpu))
for name, val in vars(args).items():
    logger.info("[PARAMS] {}: {}".format(name, val))

def average_gradients(tower_grads):
    average_grads = []
    ## grad_and_vars：((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        ## average over the `tower` dimension
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        ## Variables are redundant because they are shared cross towers
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def compute_accuracy(y_pred, y_true):
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

tf.reset_default_graph()
with tf.Graph().as_default(), tf.device('/cpu:0'):
    try:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        add_global_step = global_step.assign_add(1)
        optimizer, lr_schedule = make_optimizer(args, global_step)
        xentropy_loss = tf.losses.sparse_softmax_cross_entropy
        ## model forward
        model = Model.model
        input_shape = [None, INPUT_SIZE, INPUT_SIZE, 3]
        anchor = tf.placeholder(tf.float32, input_shape, name="anchor")
        pos = tf.placeholder(tf.float32, input_shape, name="positive")
        neg = tf.placeholder(tf.float32, input_shape, name="negative")
        label = tf.placeholder(tf.int32, [None,])
        anchor_splits = tf.split(anchor, num_or_size_splits=num_gpu, axis=0)
        pos_splits = tf.split(pos, num_or_size_splits=num_gpu, axis=0)
        neg_splits = tf.split(neg, num_or_size_splits=num_gpu, axis=0)
        label_splits = tf.split(label, num_or_size_splits=num_gpu, axis=0)
        tower_grads, tower_loss, tower_accu = [], [], []
        update_ops = []
        ## Loop over all GPUs and construct their own computation graph
        for i, gpu in enumerate(GPUs):
            with tf.device("/gpu:%d" % gpu):
                ## reuse_vars = True
                with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
                    anchor_feature, anchor_logits = model(anchor_splits[i], is_training=True)
                    pos_feature, pos_logits = model(pos_splits[i], is_training=True)
                    neg_feature, neg_logits = model(neg_splits[i], is_training=True)
                    ## loss
                    feature_loss = triplet_loss(
                        anchor_feature,
                        pos_feature,
                        neg_feature,
                        args.margin
                    )
                    logits_loss = 0.5 * (xentropy_loss(label_splits[i], anchor_logits) + xentropy_loss(label_splits[i], neg_logits))
                    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
                    total_loss = args.omega * feature_loss + (1 - args.omega) * logits_loss + args.weight_decay * regularization_loss
                    ## accuracy
                    anchor_accuracy = compute_accuracy(anchor_logits, label_splits[i])
                    pos_accuracy = compute_accuracy(pos_logits, label_splits[i])
                    neg_accuracy = compute_accuracy(neg_logits, label_splits[i])
                    total_accuracy = (anchor_accuracy + pos_accuracy + neg_accuracy) / 3.0
                    ## model summary
                    # if i == 0: Model.model_summary()

                    ## collect grads
                    grads = optimizer.compute_gradients(
                        total_loss,
                        var_list=tf.trainable_variables()
                    )
                    tower_grads.append([x for x in grads if x[0] is not None])
                    tower_loss.append(total_loss)
                    tower_accu.append(total_accuracy)
                    ## collect moving_mean and moving_variance in batch_norm
                    update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                    ## retain summaries from the last tower
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        loss_avg = tf.reduce_mean(tower_loss, axis=0)
        accu_avg = tf.reduce_mean(tower_accu, axis=0)
        grads_avg = average_gradients(tower_grads)
        grads_op = optimizer.apply_gradients(grads_avg, global_step)
        ## use moving averages of trainable variables to boost generalization power
        variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        ## tf.summary, feature_loss & logits_loss & train_accuracy are tracked from the last tower
        summaries.append(tf.summary.scalar("learning_rate", lr_schedule))
        summaries.append(tf.summary.scalar("loss/all_loss", loss_avg))
        summaries.append(tf.summary.scalar("loss/feature_loss", feature_loss))
        summaries.append(tf.summary.scalar("loss/logits_loss", logits_loss))
        summaries.append(tf.summary.scalar("train_accuracy", total_accuracy))
        with tf.control_dependencies(update_ops):
            train_op = tf.group(grads_op, variables_averages_op)
    except Exception as e:
        logger.error("Initializing error: {}".format(e))
        sys.exit(0)

    arguments = {}
    arguments.update(vars(args))
    arguments["itr"] = 0
    logger.info("Loading train_data ......")
    global_batch_size = args.batch_size * num_gpu
    train_data = load_train_triplet(global_batch_size)
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
    restorer = Model.make_variable_restorer()
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True),
            allow_soft_placement=True,
            log_device_placement=False
        )) as sess:
        ## initialize weights
        sess.run(init)
        restorer.restore(sess, args.ckpt)
        logger.info("Restored {}".format(args.ckpt))
        ## merge summary
        merge_summary_op = tf.summary.merge(summaries)
        summary_writer = tf.summary.FileWriter(
            "{}/summary".format(args.out_dir), 
            sess.graph
        )
        ## train
        tf.train.start_queue_runners(sess=sess)
        for itr in range(start_itr, max_itr+1):
            batch_data = sess.run(next_train_data)
            _, batch_tl, batch_xl, batch_accu, add, lr = sess.run(
                fetches=[
                    train_op,
                    feature_loss, logits_loss,
                    accu_avg,
                    add_global_step, lr_schedule
                ],
                feed_dict={
                    anchor: batch_data[0],
                    pos: batch_data[1],
                    neg: batch_data[2],
                    label: batch_data[3]
                }
            )
            assert not (np.isnan(batch_tl) or np.isnan(batch_xl)), 'Model diverged with loss = NaN'
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
                ## summary writing
                summary_str = sess.run(
                    merge_summary_op,
                    feed_dict={
                        anchor: batch_data[0],
                        pos: batch_data[1],
                        neg: batch_data[2],
                        label: batch_data[3]
                    }
                )
                summary_writer.add_summary(summary_str, itr)

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


'''
python trainer.py
'''
