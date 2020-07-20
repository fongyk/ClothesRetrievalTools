import cv2
import os
import json
import numpy as np
import shutil

from filter_bbox import get_filtered

TARGET = "validation" ## validation or test

ENV_FOLDER = "/data6/fong/maskrcnn_env/maskrcnn-benchmark-sibling"
INFO_FOLDER = os.path.join(ENV_FOLDER, "output/predict")
if TARGET == "validation":
    IMAGE_FOLDER = "/data6/fong/DeepFashion/unzip/validation/image"
else:
    IMAGE_FOLDER = os.path.join("/data6/fong/DeepFashion/unzip/test/test/image")
SAVE_FOLDER = os.path.join("/data6/fong/DeepFashion/crop/", TARGET)


def load_json(json_file):
    with open(json_file, "r") as fr:
        return json.load(fr)

def crop_by_bbox(json_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    info = load_json(json_file)
    bboxes = info["bbox"]
    image_name = info["image"]
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    print("cropping {}".format(image_path))
    if len(bboxes) == 0:
        dst_path = os.path.join(save_folder, image_name.split(".")[0]+"-0.jpg")
        try:
            shutil.copy(image_path, dst_path)
        except:
            print("can not copy {}".format(image_path))
    else:
        bboxes = get_filtered(bboxes)[0]
        image = cv2.imread(image_path)
        for bid, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = list(map(int, bbox))
            region = image[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(save_folder, image_name.split(".")[0]+"-{}.jpg".format(bid)), region)

def crop(info_folder, source, save_folder):
    dir_folder = os.path.join(info_folder, source)
    json_files = os.listdir(dir_folder)
    for json_file in json_files:
        crop_by_bbox(os.path.join(dir_folder, json_file), os.path.join(save_folder, source))

if __name__ == "__main__":
    if TARGET == "validation":
        crop(INFO_FOLDER, "query", SAVE_FOLDER)
        crop(INFO_FOLDER, "gallery", SAVE_FOLDER)
    else:
        crop(INFO_FOLDER, "consumer", SAVE_FOLDER)
        crop(INFO_FOLDER, "shop", SAVE_FOLDER)