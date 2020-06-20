'''
Map image to a fine-grained class according to pair_id and style,
so as to sample training pairs.

Images are arranged as follows:
    #category_1
        ##pair_id_0
            ###style_0
                ####shop
                    xxxx.jpg
                    ......
                ####user
                    xxxx.jpg
                    ......
            ###style_1
            ......
        ##pair_id_1
        ......
    #category_2
    ......
    #category_13
'''

import os
import glob
import yaml
from collections import defaultdict

from ..config.defaults import TRAIN_PATH

def load_yaml(yaml_file):
    with open(yaml_file, "r") as fr:
        info = yaml.load(fr, Loader=yaml.FullLoader)
    return info

def write_yaml(yaml_path, info):
    with open(yaml_path, "w") as fw:
        yaml.dump(info, fw)

def map_style_to_finegrained_class(dataset_path=TRAIN_PATH):
    styles = glob.glob(os.path.join(dataset_path, "*", "*", "style*"))
    styles.sort()
    mapper = defaultdict(dict)
    for fine_cls, style in enumerate(styles, 1):
        style_split = style.strip().split("/")
        coarse_cls, pair_id, style_name = int(style_split[-3]), style_split[-2], style_split[-1]
        if not mapper[coarse_cls].get(pair_id):
            mapper[coarse_cls][pair_id] = {}
        mapper[coarse_cls][pair_id][style_name] = fine_cls
    mapper = dict(mapper)
    write_yaml("base/data/finegrain_mapper.yaml", mapper)

def get_finegrained_mapper():
    mapper = load_yaml("base/data/finegrain_mapper.yaml")
    return mapper
