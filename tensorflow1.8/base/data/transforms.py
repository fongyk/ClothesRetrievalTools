from ..config.defaults import NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE, NUM_CLASSES
import tensorflow as tf

def normalize(x, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    x = (x - mean) / std
    return x

def denormalize(x, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    x = x * std + mean
    return x

def process_image(x):
    '''
    x: image_path
    '''
    x = tf.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize_images(x, [INPUT_SIZE, INPUT_SIZE])
    # x = tf.image.random_flip_left_right(x)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    return x


def preprocess_triplet(*triplet):
    '''
    x: triplet, (anchor, pos, neg, label), paths to three images & category

    one-hot encoding for CategoricalCrossentropy: (batch_size, #class)
    integer for SparseCategoricalCrossentropy: (batch_size,)
    '''
    
    x0, x1, x2, y = triplet
    x0, x1, x2 = process_image(x0), process_image(x1), process_image(x2)
    y = tf.convert_to_tensor(y)
    # y = tf.one_hot(y, depth=NUM_CLASSES)
    return x0, x1, x2, y


def preprocess_pair(x, y):
    '''
    x: paths to image
    y: label / image_name

    one-hot encoding for CategoricalCrossentropy: (batch_size, #class)
    integer for SparseCategoricalCrossentropy: (batch_size,)
    '''

    x = process_image(x)
    y = tf.convert_to_tensor(y)
    # y = tf.one_hot(y, depth=NUM_CLASSES)
    return x, y

def preprocess_pair_with_multi_targets(x, y):
    '''
    x: paths to image
    y: {name: target}
    '''

    x = process_image(x)
    y = {k: tf.convert_to_tensor(v) for k, v in y.items()}
    return x, y