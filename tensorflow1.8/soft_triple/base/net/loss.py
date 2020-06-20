import tensorflow as tf

def triplet_loss(anchor, pos, neg, margin):
    pos_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)
    neg_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
    loss       = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

from ..config.defaults import NUM_FINECLASSES, NUM_CENTERS, HEAD_CLS
def soft_triple_loss(
    labels,
    logits,
    large_centers,
    dim_feature=2048,
    num_classes=NUM_FINECLASSES,
    num_centers=NUM_CENTERS,
    p_lambda=20.0,
    p_gamma=0.1,
    p_delta=0.01,
    p_tau=0.2
):
    ## get center mask
    def get_center_mask():
        '''
        Suppose: num_classes=3, num_centers=2
        Return:
            array([[1., 1., 0., 0., 0., 0.],
                   [1., 1., 0., 0., 0., 0.],
                   [0., 0., 1., 1., 0., 0.],
                   [0., 0., 1., 1., 0., 0.],
                   [0., 0., 0., 0., 1., 1.],
                   [0., 0., 0., 0., 1., 1.]], dtype=float32)
        '''
        mask = tf.range(num_classes, dtype=tf.int32)
        mask = tf.one_hot(mask, depth=num_classes, dtype=tf.float32) # C x C
        mask = tf.expand_dims(mask, 1) # C x 1 x C
        mask = tf.tile(mask, [1, num_centers, 1]) # C x K x C
        mask = tf.reshape(mask, [-1, num_classes]) # CK x C
        mask = tf.expand_dims(mask, 2) # CK x C x 1
        mask = tf.tile(mask, [1, 1, num_centers]) # CK x C x K
        mask = tf.reshape(mask, [num_classes*num_centers, -1])
        return mask

    ## get logits
    # batch_size = tf.shape(features)[0]
    # large_centers = tf.get_variable(
    #     name=HEAD_CLS,
    #     shape=[num_classes*num_centers, dim_feature],
    #     dtype=tf.float32,
    #     initializer=tf.contrib.layers.xavier_initializer(uniform=True),
    #     trainable=True
    # )
    # large_centers = tf.nn.l2_normalize(large_centers, axis=-1)
    # large_logits = tf.matmul(features, large_centers, transpose_b=True)
    # large_logits = tf.reshape(large_logits, [batch_size, num_centers, num_classes])
    # exp_logits = tf.exp((1.0 / p_gamma) * large_logits)
    # sum_exp_logits = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
    # coeff_logits = exp_logits / sum_exp_logits
    # large_logits = tf.multiply(large_logits, coeff_logits)
    # logits = tf.reduce_sum(large_logits, axis=1, keepdims=False)

    ## get label map
    target = tf.reshape(labels, [-1])
    target = tf.cast(target, tf.int32)
    target_map = tf.one_hot(target, depth=num_classes, dtype=tf.float32)
    delta_map = p_delta * target_map
    post_logits = logits - delta_map
    post_logits = p_lambda * post_logits

    ## get cross-entropy loss
    loss_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=post_logits, labels=target_map)
    loss_xentropy = tf.reduce_mean(loss_xentropy, name="xentropy_loss")

    ## get regularizer
    sim_large_centers = tf.matmul(large_centers, large_centers, transpose_b=True)
    dist_large_centers = tf.sqrt(tf.abs(2.0 - 2.0 * sim_large_centers) + 1e-10)
    center_mask = get_center_mask()
    dist_large_centers = tf.multiply(dist_large_centers, center_mask)
    dist_mask = tf.ones_like(dist_large_centers, dtype=tf.float32) - tf.eye(num_classes*num_centers, dtype=tf.float32)
    dist_large_centers = tf.multiply(dist_large_centers, dist_mask) / 2.0
    reg = p_tau * tf.reduce_sum(dist_large_centers)
    loss_reg = reg / (num_classes * num_centers * (num_centers - 1.0))

    return loss_xentropy + loss_reg
