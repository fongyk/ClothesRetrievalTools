import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.structures.image_list import to_image_list

import os
import random
import linecache
import json
from PIL import Image

min_keypoints_per_image = 1
num_keypoints_per_class = [25, 33, 31, 39, 15, 15, 10, 14, 8, 29, 37, 19, 19]
keypoint_offset_per_class = [sum(num_keypoints_per_class[:i]) for i in range(13)]
KYEPOINT_NUM = 294

QUADRUPLE_NUM = 200000

ann_file = '/data6/fong/DeepFashion/unzip/train/annos'
quadruple_path = '/data6/fong/DeepFashion/quadruple_crop_with_label_to_category'

from .dataloader import IMG_TRANSFORMS

def loadjson(json_file):
    with open(json_file, 'r') as fr:
        ann = json.load(fr)
        return ann

def getlabels(ann):
    labels = []
    for k in list(ann.keys()):
        if k.startswith('item'):
            labels.append(ann[k]['category_id'])
    labels = torch.as_tensor(labels)
    return labels

def getbbox(ann):
    boxes = []
    for k in list(ann.keys()):
        if k.startswith('item'):
            boxes.append(ann[k]['bounding_box'])
    boxes = torch.as_tensor(boxes).reshape(-1, 4)
    return boxes

def getstyles(ann):
    styles = []
    for k in list(ann.keys()):
        if k.startswith('item'):
            styles.append(ann[k]['style'])
    styles = torch.as_tensor(styles)
    return styles

def getseg(ann):
    masks = []
    for k in list(ann.keys()):
        if k.startswith('item'):
            masks.append(ann[k]['segmentation'])
    return masks

def getkeypoints(ann):
    keypoints = []
    for k in list(ann.keys()):
        if k.startswith('item'):
            template = [0] * 3 * KYEPOINT_NUM
            kps = ann[k]['landmarks']
            label = ann[k]['category_id']
            offset = keypoint_offset_per_class[label - 1] * 3
            template[offset:offset+len(kps)] = kps
            keypoints.append(template)
    return keypoints

class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, item_num=QUADRUPLE_NUM, transforms=IMG_TRANSFORMS):
        self.ann_file = ann_file
        self.quadruple_path = quadruple_path
        self.quadruple_num = QUADRUPLE_NUM
        self.train_quadruple_num = item_num
        self.transforms = transforms

    @staticmethod
    def filltarget(ann, size):
        labels = getlabels(ann)
        boxes = getbbox(ann)
        target = BoxList(boxes, size, mode="xyxy")
        target.add_field("labels", labels)
        styles = getstyles(ann)
        target.add_field("styles", styles)
        masks = getseg(ann)
        masks = SegmentationMask(masks, size, mode='poly')
        target.add_field("masks", masks)
        keypoints = getkeypoints(ann)
        keypoints = PersonKeypoints(keypoints, size)
        target.add_field("keypoints", keypoints)
        target = target.clip_to_image(remove_empty=True)
        return target

    def __getitem__(self, idx):
        quadruple_file = os.path.join(self.quadruple_path, 
                                "{:06d}".format(idx % self.quadruple_num + 1))
        images = []
        labels = []
        with open(quadruple_file, 'r') as fr:
            imgs = fr.readline().strip().split(' ')
            ## quadruple: shop_anchor user_anchor shop_negative user_negative
            for img in imgs:
                ## img: /path/to/image_id.category.jpg
                category = int(img.split('.')[-2])
                img = Image.open(img).convert('RGB')
                if self.transforms:
                    img = self.transforms(img)
                images.append(img)
                labels.append(category)
        return images[0], labels[0], images[1], labels[1], images[2], labels[2], images[3], labels[3]

    def __len__(self):
        return self.train_quadruple_num
