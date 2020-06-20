import tensorflow as tf

def make_optimizer(args, global_step):
    if args.scheduler == "exp":
        lr_schedule = tf.train.exponential_decay(
            learning_rate=args.learning_rate,
            global_step=global_step,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True,
            name="ExponentialDecay"
        )
    elif args.scheduler == "piece":
        boundaries = args.lr_milestones
        values = [args.learning_rate*(args.lr_gamma**step) for step in range(len(boundaries)+1)]
        lr_schedule = tf.train.piece_constant(
            global_step=global_step,
            boundaries=boundaries, 
            values=values, 
            name="PiecewiseConstantDecay"
        )
    else:
        raise NotImplementedError("choose exp or piece scheduler.")

    if args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(lr_schedule)
    elif args.optimizer == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(lr_schedule)
    else:
        raise NotImplementedError("Choose Adam or SGD")
    
    return optimizer, lr_schedule