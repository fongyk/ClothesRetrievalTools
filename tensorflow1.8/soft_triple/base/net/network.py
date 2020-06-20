import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

from ..config.defaults import NUM_CLASSES, NUM_FINECLASSES, NUM_CENTERS, INPUT_SIZE, HEAD_FT, HEAD_CLS

class Model(object):
    @staticmethod
    def model(inputs, dim_feature=2048, gamma=0.1, is_training=True):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(
                    inputs,
                    num_classes=None,
                    is_training=is_training,
                    global_pool=True,
                )
            net = tf.squeeze(net, axis=[1,2])
            feature = tf.nn.l2_normalize(net, axis=1)
            end_points[HEAD_FT] = feature
            # logits = slim.fully_connected(
            #     net,
            #     num_outputs=NUM_CLASSES,
            #     activation_fn=None,
            #     scope=HEAD_CLS
            # )
            # end_points[HEAD_CLS] = logits
            # return feature, logits
            batch_size = tf.shape(feature)[0]
            large_centers = tf.get_variable(
                name=HEAD_CLS,
                shape=[NUM_FINECLASSES*NUM_CENTERS, dim_feature],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                trainable=True
            )
            large_centers = tf.nn.l2_normalize(large_centers, axis=-1)
            large_logits = tf.matmul(feature, large_centers, transpose_b=True)
            large_logits = tf.reshape(large_logits, [batch_size, NUM_CENTERS, NUM_FINECLASSES])
            exp_logits = tf.exp((1.0 / gamma) * large_logits)
            sum_exp_logits = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
            coeff_logits = exp_logits / sum_exp_logits
            large_logits = tf.multiply(large_logits, coeff_logits)
            logits = tf.reduce_sum(large_logits, axis=1, keepdims=False)
            end_points[HEAD_CLS] = logits
            return feature, logits, large_centers

    @staticmethod
    def make_variable_restorer(exclude_scope=[HEAD_CLS, "global_step"]):
        '''
        restore pretrained weights.
        '''
        def get_variables_to_restore():
            variables_to_train = []
            for var in tf.trainable_variables():
                excluded = False
                for exclusion in exclude_scope:
                    if var.op.name.startswith(exclusion) or var.op.name.endswith("ExponentialMovingAverage"):
                        excluded = True
                if not excluded:
                    variables_to_train.append(var)
            return variables_to_train
        variables_to_restore = get_variables_to_restore()
        restorer = tf.train.Saver(var_list=variables_to_restore)
        return restorer

    @staticmethod
    def model_summary():
        print("\n")
        print("="*30 + "Model Summary" + "="*30)
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        print("="*65 + "\n")
