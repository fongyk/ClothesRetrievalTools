#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

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

@tf.function
def val_step(
    model,
    val_acc,
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
def train_step(
    siamese_model,
    optimizer,
    train_acc,
    omega,
    triplet_loss, 
    xentropy_loss,
    anchor, 
    pos, 
    neg, 
    label, 
):
    with tf.GradientTape() as tape:
        anchor_out, pos_out, neg_out = siamese_model([anchor, pos, neg])
        anchor_feature, anchor_logits = anchor_out
        pos_feature, pos_logits = pos_out
        neg_feature, neg_logits = neg_out
        loss_feature = triplet_loss(label, [anchor_feature, pos_feature, neg_feature])
        loss_logits = 0.5 * (xentropy_loss(label, anchor_logits) + xentropy_loss(label, neg_logits))
        loss_total = [omega * loss_feature, (1 - omega) * loss_logits]

    gradients = tape.gradient(loss_total, siamese_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, siamese_model.trainable_variables))

    train_acc.update_state(label, anchor_logits)
    train_acc.update_state(label, neg_logits)

    return dict(
        loss_total=sum(loss_total), 
        loss_feature=loss_feature, 
        loss_logits=loss_logits
    )

def train(args):
    logger = set_logger("trainer", args.out_dir)
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))
    try:
        model = nets[args.net]()
        siamese_model = ntk.build_threestream_siamese_network(model)
        siamese_model.summary()
        keras.utils.plot_model(siamese_model, '{}/siamese_model.png'.format(args.out_dir), show_shapes=True)
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return
    logger.info("Training {}.".format(args.net))

    keras.backend.set_learning_phase(True)

    loss_objects = [
        net_loss.TripletLoss(args.margin),
        keras.losses.SparseCategoricalCrossentropy(),
    ]
    optimizer = make_optimizer(args)
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

    arguments = {}
    arguments.update(vars(args))
    arguments["itr"] = 0

    logger.info("Loading train_data ......")
    train_data = load_train_triplet(args.batch_size, args.train_num)
    ## split train and validation set
    val_data = train_data.take(args.val_num)
    train_data = train_data.skip(args.val_num)
    ## train model
    logger.info("Start training ......")
    meters = MetricLogger(delimiter=", ")
    max_itr = args.train_num
    start_itr = arguments["itr"] + 1
    itr_start_time = time.time()
    training_start_time = time.time()
    for itr, batch_data in enumerate(train_data, start_itr):
        loss_dict = train_step(
            siamese_model, 
            optimizer,
            train_acc,
            args.omega, 
            loss_objects[0], 
            loss_objects[1], 
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
                val_step(model, val_acc, *batch_data)
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


def fit(args):
    logger = set_logger("trainer", args.out_dir)
    for name, val in vars(args).items():
        logger.info("[PARAMS] {}: {}".format(name, val))
    try:
        base_model = nets[args.net]()
    except Exception as e:
        logger.error("Initialize {} error: {}".format(args.net, e))
        return
    logger.info("Training {}.".format(args.net))

    logger.info("Loading train_data ......")
    train_data = load_train_pair_with_two_targets(args.batch_size, args.train_num)
    val_data = train_data.take(args.val_num)
    train_data = train_data.skip(args.val_num)
    # images, labels = load_train_pair_from_numpy(args.train_num)

    keras.backend.set_learning_phase(True)
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
    ## compile model
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
    
    callbacks = make_callbacks(args)
    model.compile(
        optimizer=optimizer, 
        loss=losses, 
        loss_weights=loss_weights,
        metrics=metrics
    )
    ## train model
    logger.info("Start training ......")
    H = model.fit(
        x=train_data.repeat(),
        validation_data=val_data,
        epochs=args.epoch_num,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    # H = model.fit(
    #     x=images,
    #     y={
    #         HEAD_FT: labels,
    #         HEAD_CLS: labels
    #     },
    #     validation_split=0.1,
    #     epochs=args.epoch_num,
    #     batch_size=args.batch_size,
    #     callbacks=callbacks,
    #     verbose=1
    # )
    
    train_acc = H.history["logits_sparse_categorical_accuracy"]
    val_acc = H.history["val_logits_sparse_categorical_accuracy"]
    logger.info("train_accuracy: {}".format(list(map(lambda e:round(e,3), train_acc))))
    logger.info("val_accuracy: {}".format(list(map(lambda e:round(e,3), val_acc))))

    model.save(os.path.join(args.out_dir, "model_final.h5"))
    # new_model = keras.models.load_model("model_final.h5")
    with open(os.path.join(args.out_dir, "arguments.json"), "w") as fw:
        json.dump(vars(args), fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Fashion2 Retrieval.')
    parser.add_argument('--net', type=str, default="vggnet", help='select vggnet/resnet')
    parser.add_argument('--train_num', type=int, default=100000, help='num of training samples to load')
    parser.add_argument('--val_num', type=int, default=1000, help='size of validation set')
    parser.add_argument('--epoch_num', type=int, default=10, help='epoch num')
    parser.add_argument('--steps_per_epoch', type=int, default=5000, help='steps_per_epoch for `fit` mode')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (note: TripletSemiHardLoss expects large batch size)')
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
    parser.add_argument('--use-fit', dest='use_fit', action='store_true', help='use `fit` mode (default: `train` mode)')
    parser.add_argument('--use-amp', dest='use_amp', action='store_true', help='use Automatic Mixed Precision')
    parser.add_argument('--use-cpu', dest='use_cpu', action='store_true', help='this is tf-gpu, use-cpu to run on cpu')
    args, _ = parser.parse_known_args()

    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

    ## TODO
    # if args.use_amp:
        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_policy(policy)
        # print('Compute dtype: %s' % policy.compute_dtype)
        # print('Variable dtype: %s' % policy.variable_dtype)
        
    if args.use_fit:
        fit(args)
    else:
        train(args)

'''GPU
CUDA_VISIBLE_DEVICES=1 python trainer.py --net=resnet -out_dir=output
'''

'''CPU
CUDA_VISIBLE_DEVICES=-1 python trainer.py --net=resnet -out_dir=output
'''