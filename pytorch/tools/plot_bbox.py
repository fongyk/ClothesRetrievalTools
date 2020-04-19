import cv2
import os
import json
import numpy as np

from filter_bbox import get_filtered

category_names = ["None",
                  "short sleeve top",
                  "long sleeve top",
                  "short sleeve outwear",
                  "long sleeve outwear",
                  "vest",
                  "sling",
                  "shorts",
                  "trousers",
                  "skirt",
                  "short sleeve dress",
                  "long sleeve dress",
                  "vest dress",
                  "sling dress"]

IMAGE_FOLDER = "/data6/fong/DeepFashion/unzip/validation/image"
ENV_FOLDER = "/data6/fong/maskrcnn_env/maskrcnn-benchmark-sibling"
QUERY_INFO_FOLDER = os.path.join(ENV_FOLDER, "output/predict/query")
GALLERY_INFO_FOLDER = os.path.join(ENV_FOLDER, "output/predict/gallery")

def load_json(json_file):
    with open(json_file, "r") as fr:
        return json.load(fr)

def plot_boungding_box(image_folder, info_folder, max_num=2000):
    font = cv2.FONT_HERSHEY_SIMPLEX
    json_files = os.listdir(info_folder)
    json_files.sort()
    image_num_with_no_bbox = 0
    for jid, json_file in enumerate(json_files):
        if jid == max_num: break
        info = load_json(os.path.join(info_folder, json_file))
        image = os.path.join(image_folder, info["image"])
        image = cv2.imread(image)
        bboxes = info["bbox"]
        scores = [round(s, 2) for s in info["scores"]]
        labels = info["labels"]
        if len(scores) == 0:
            image_num_with_no_bbox += 1
        else:
            assert len(info["feature"]) == len(bboxes), "#bbox not compatible with #feature"
        bboxes, scores, labels = get_filtered(bboxes, scores, labels)
        # print(scores)
        # print("features:", info["feature"], sep='\n')
        for bid, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = list(map(int, bbox))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.putText(image, str(scores[bid]), (x1+20, y1+20), font, 0.6, (255,0,0), 2)
            cv2.putText(image, category_names[labels[bid]], (x1+70, y1+20), font, 0.6, (0,0,255), 2)
        cv2.imwrite("bbox_result/{}".format(info["image"]), image)
    print("{}/{} images without bbox.".format(image_num_with_no_bbox, min(max_num, len(json_files))))


if __name__ == "__main__":
    plot_boungding_box(IMAGE_FOLDER, QUERY_INFO_FOLDER)
    plot_boungding_box(IMAGE_FOLDER, GALLERY_INFO_FOLDER)

    # info = load_json(os.path.join(GALLERY_INFO_FOLDER, "010933.json"))
    # bboxes = info["bbox"]
    # scores = info["scores"]
    # labels = info["labels"]
    # print(scores)
    # print(labels)
    # bboxes, scores, labels = get_filtered(bboxes, scores, labels)
    # print(scores)
    # print(labels)
