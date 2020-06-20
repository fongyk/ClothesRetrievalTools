import os
import glob
import random
import csv
import numpy as np

from ..config.defaults import TRAIN_PAIRS, TRAIN_PATH, HEAD_FT, HEAD_CLS
from .transforms import preprocess_triplet, preprocess_pair, process_image, preprocess_pair_with_multi_targets
from .finegrain_mapper import get_finegrained_mapper

import tensorflow as tf

def sample_triplet_with_category(
    train_root=TRAIN_PATH,
    num_pairs=TRAIN_PAIRS,
    verbose=True
):
    category_ids = os.listdir(train_root)
    category_ids.sort()
    n = 0
    finished = False
    triplets = []
    while not finished:
        for cid in category_ids:
            pair_ids = os.listdir(os.path.join(train_root, cid))
            if len(pair_ids) < 2: continue
            for pid in pair_ids:
                ## sample shop_anchor and user_anchor
                styles = os.listdir(os.path.join(train_root, cid, pid))
                if 'style_1' not in styles: continue
                while True:
                    style_anchor = random.choice(styles)
                    if style_anchor != 'style_0':
                        break
                shop_user = os.listdir(os.path.join(train_root, cid, pid, style_anchor))
                if len(shop_user) < 2: continue
                shop_anchor = random.choice(glob.glob(os.path.join(train_root, cid, pid, style_anchor, 'shop', '*.jpg')))
                user_anchor = random.choice(glob.glob(os.path.join(train_root, cid, pid, style_anchor, 'user', '*.jpg')))
                ## sample negative
                while True:
                    pid_negative = random.choice(pair_ids)
                    if pid_negative != pid:
                        break
                negatives = glob.glob(os.path.join(train_root, cid, pid_negative, 'style_[1-9]', '*', '*.jpg'))
                if len(negatives) < 1: continue
                negative = random.choice(negatives)
                if not verbose:
                    shop_anchor = shop_anchor.split('/')[-1]
                    user_anchor = user_anchor.split('/')[-1]
                    negative = negative.split('/')[-1]
                n += 1
                triplets.append([shop_anchor, user_anchor, negative, cid])
                if n > num_pairs:
                    finished = True
                    break
    with open("base/data/triplets.csv", "w") as fw:
        fw_csv = csv.writer(fw)
        fw_csv.writerows(triplets)


def load_train_triplet(batch_size=1, load_num=None):
    '''
    `Prefetch` overlaps the preprocessing and model execution of a training step.
    While the model is executing training step s, the input pipeline is reading the data for step s+1.

    return: (anchor, pos, neg, label)
    '''
    anchors_list, positives_list, negatives_list, labels = [], [], [], []
    with open("base/data/triplets.csv", "r") as fr:
        triplets = list(csv.reader(fr))
        if load_num is not None:
            assert load_num > 0
            triplets = triplets[:load_num]
        for a, p, n, l in triplets:
            anchors_list.append(a)
            positives_list.append(p)
            negatives_list.append(n)
            labels.append(int(l))
    train_data = tf.data.Dataset.from_tensor_slices((anchors_list, positives_list, negatives_list, labels))
    train_data = train_data.repeat() ## samples_num = max_itr * batch_size
    train_data = train_data.shuffle(1000).map(preprocess_triplet).batch(batch_size).prefetch(tf.contrib.data.AUTOTUNE)
    return train_data

def load_train_pair(batch_size=1, load_num=None):
    '''
    return: pair (image, label)
    label: fine-grained category
    '''
    def get_finegrained_cls(image_path):
        image_info = image_path.strip().split("/")
        coarse_cls, pair_id, style_name = int(image_info[-5]), image_info[-4], image_info[-3]
        return mapper[coarse_cls][pair_id][style_name]
    images, labels = [], []
    mapper = get_finegrained_mapper()
    with open("base/data/triplets.csv", "r") as fr:
        triplets = list(csv.reader(fr))
        if load_num is not None:
            assert load_num > 0 and load_num <= len(triplets)
            triplets = triplets[:load_num]
        for a, p, n, l in list(triplets):
            a_fine_cls, n_fine_cls = get_finegrained_cls(a), get_finegrained_cls(n)
            images.extend([a, n])
            # labels.extend([a_fine_cls, n_fine_cls])
            labels.extend([l, l])
    train_data = tf.data.Dataset.from_tensor_slices((images, labels))
    train_data = train_data.repeat() 
    train_data = train_data.shuffle(1000).map(preprocess_pair).batch(batch_size).prefetch(tf.contrib.data.AUTOTUNE)
    return train_data

def load_train_pair_with_two_targets(batch_size=1, load_num=None):
    '''
    return: pair (image, target_1, target_2)
    target_1: for fine-grained embedding loss
    target_2: for coarse-grained cross-entropy loss
    '''
    def get_finegrained_cls(image_path):
        image_info = image_path.strip().split("/")
        coarse_cls, pair_id, style_name = int(image_info[-5]), image_info[-4], image_info[-3]
        return mapper[coarse_cls][pair_id][style_name]

    images, target_1, target_2 = [], [], []
    mapper = get_finegrained_mapper()
    with open("base/data/triplets.csv", "r") as fr:
        triplets = list(csv.reader(fr))
        if load_num is not None:
            assert load_num > 0 and load_num <= len(triplets)
            triplets = triplets[:load_num]
        for a, p, n, l in list(triplets):
            coarse_cls = int(l)
            a_fine_cls, p_fine_cls = get_finegrained_cls(a), get_finegrained_cls(p)
            images.extend([a, p]) ## ensure anchor-positive pair in mini-batch
            target_1.extend([a_fine_cls, p_fine_cls])
            target_2.extend([coarse_cls, coarse_cls])

    train_data = tf.data.Dataset.from_tensor_slices((
        images,
        {HEAD_FT: target_1, HEAD_CLS: target_2}
        ))
    train_data = train_data.map(preprocess_pair_with_multi_targets)
    train_data = train_data.batch(batch_size).shuffle(1000).prefetch(tf.contrib.data.AUTOTUNE)
    return train_data

def load_train_pair_from_numpy(load_num=None):
    '''
    WARNING: very slow.
    Open and read the whole dataset all at once.

    return: images, labels
    '''
    images, labels = [], []
    with open("base/data/triplets.csv", "r") as fr:
        triplets = list(csv.reader(fr))
        if load_num is not None:
            assert load_num > 0 and load_num <= len(triplets)
            triplets = triplets[:load_num]
        for a, p, n, l in list(triplets):
            image = process_image(a).numpy()
            images.append(image)
            labels.append(int(l)) ## without one-hot encoding
    images = np.array(images)
    labels = np.array(labels)
    shuffle = np.random.permutation(len(images))
    images = images[shuffle]
    labels = labels[shuffle]
    return images, labels
