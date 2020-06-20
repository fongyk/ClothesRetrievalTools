import os
import math

from ..config.defaults import TRAIN_PAIRS, TRAIN_PATH
from .transforms import process_image, preprocess_pair

import tensorflow as tf

def load_from_list(path_to_list):
    with open(path_to_list, "r") as fr:
        images = fr.readlines()
        images = list(map(lambda image:image.strip(), images))
        names = list(map(lambda image:image.split("/")[-1], images))
    return images, names

def load_test_data(path_to_list, batch_size=1):
    '''
    `repeat()` involves repetitive computation for list-tail images,
    but it fixes error caused by `tf.split: Number of ways to split should evenly divide the split dim`.
    '''
    images, names = load_from_list(path_to_list)
    test_data = tf.data.Dataset.from_tensor_slices((images, names)).repeat()
    test_data = test_data.map(preprocess_pair).batch(batch_size).prefetch(tf.contrib.data.AUTOTUNE)
    return test_data, int(math.ceil(len(images)/batch_size))