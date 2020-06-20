import tensorflow as tf

def triplet_loss(anchor, pos, neg, margin):
    pos_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)    
    neg_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)    
    loss       = tf.reduce_mean(tf.maximum(basic_loss, 0.0))               
    return loss
