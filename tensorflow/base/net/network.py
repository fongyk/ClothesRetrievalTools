import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config.defaults import NUM_CLASSES, INPUT_SIZE, HEAD_FT, HEAD_CLS

class BaseNet(keras.Model):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.norm = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name=HEAD_FT)
        self.pred = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype=tf.float32, name=HEAD_CLS)

    ## ToDo: Remove embedder, just use `call`
    ## `plot_model` gives a bit different structure with `embedder` and `call`
    def embedder(self, inputs):
        x = self.backbone(inputs)
        x = self.pool(x)
        feature = self.norm(x)
        logits = self.pred(x)
        return feature, logits

    def call(self, x, training=True):
        x = self.backbone(x)
        x = self.pool(x)
        feature = self.norm(x)
        # if not training:
        #     return feature
        logits = self.pred(x)
        return feature, logits

class ResNet(BaseNet):
    def __init__(self):
        super(ResNet, self).__init__()
        self.backbone = keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(INPUT_SIZE, INPUT_SIZE,3)
                    )

class VGGNet(BaseNet):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.backbone = keras.applications.VGG16(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(INPUT_SIZE, INPUT_SIZE,3)
                    )

def build_threestream_siamese_network(base_model):
    input_shape = (INPUT_SIZE, INPUT_SIZE, 3)
    anchor_in    = keras.layers.Input(input_shape, name="anchor")
    anchor_out   = base_model(anchor_in)
    pos_in       = keras.layers.Input(input_shape, name="pos")
    pos_out      = base_model(pos_in) 
    neg_in       = keras.layers.Input(input_shape, name="neg")
    neg_out      = base_model(neg_in)

    model = keras.Model(
        inputs=[anchor_in, pos_in, neg_in], 
        outputs=[anchor_out, pos_out, neg_out]
    )

    return model