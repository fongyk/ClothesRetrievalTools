import tensorflow as tf
import tensorflow.keras.losses as losses

import tensorflow_addons as tfa

'''
https://www.tensorflow.org/addons/api_docs/python/tfa/losses/TripletSemiHardLoss

default margin: 1.0
'''
TripletSemiHardLoss = tfa.losses.TripletSemiHardLoss

class TripletLoss(losses.Loss):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor, pos, neg = y_pred[0], y_pred[1], y_pred[2]
        pos_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)    
        neg_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)    
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)    
        loss       = tf.reduce_sum(tf.maximum(basic_loss, 0.0))               
        return loss
