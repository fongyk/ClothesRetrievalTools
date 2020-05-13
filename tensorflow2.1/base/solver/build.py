import tensorflow as tf
from tensorflow import keras

import os

## for train
def make_optimizer(args):
    if args.scheduler == "exp":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True,
            name="ExponentialDecay"
        )
    elif args.scheduler == "piece":
        boundaries = args.lr_milestones
        values = [args.learning_rate*(args.lr_gamma**step) for step in range(len(boundaries)+1)]
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
             boundaries=boundaries, 
             values=values, 
             name="PiecewiseConstantDecay"
        )
    else:
        raise NotImplementedError("choose exp or piece scheduler.")

    if args.optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif args.optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    else:
        raise NotImplementedError("Choose Adam or SGD")
    
    return optimizer


## for fit
def schedule(epoch, lr, lr_gamma, epoch_num):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    LR_SCHEDULE = [
        # (epoch to start, decay ratio)
        (int(epoch_num*0.5), lr_gamma), 
        (int(epoch_num*0.7), lr_gamma), 
        (int(epoch_num*0.9), lr_gamma)
    ]
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1] * lr
    return lr

class LearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.
    """
    def __init__(self, schedule, lr_gamma, epoch_num):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.lr_gamma = lr_gamma
        self.epoch_num = epoch_num

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.lr_gamma, self.epoch_num)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch {}/{}: Learning rate is {:.7f}.'.format(
                epoch+1, 
                self.epoch_num,
                scheduled_lr
            )
        )

def make_callbacks(args):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, 'callback_model_{epoch:04d}'),
            save_weights_only=True,
            verbose=1,
            save_freq='epoch'
            ),
        keras.callbacks.TensorBoard(
            log_dir=args.out_dir,
            update_freq='epoch',
            ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=3,
            verbose=1
            ),
        LearningRateScheduler(schedule, args.lr_gamma, args.epoch_num)
    ]
    return callbacks