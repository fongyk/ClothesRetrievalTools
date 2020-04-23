import numpy as np
import os
import json
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Process, Manager, Queue
import time

from utils.filter_bbox import get_filtered

TEST = False

RERANK = 0

CLASS_NUM = 14

ROOT_FOLDER = "/data6/fong/maskrcnn_env"
ENV_FOLDER = "maskrcnn-benchmark-sibling"
if TEST:
    DATABASE_FOLDER = os.path.join(ROOT_FOLDER, ENV_FOLDER, "output/predict/shop")
    QUERY_FOLDER = os.path.join(ROOT_FOLDER, ENV_FOLDER, "output/predict/consumer")
else:
    DATABASE_FOLDER = os.path.join(ROOT_FOLDER, ENV_FOLDER, "output/predict/gallery")
    QUERY_FOLDER = os.path.join(ROOT_FOLDER, ENV_FOLDER, "output/predict/query")

FEAT_FOLDER = "feat"

SAVE_DIR = "/data6/fong/DeepFashion/code/utils"

TEMPLATE = {
    "query_image_id" : 0, 
    "query_bbox" : [0,0,0,0], 
    "query_cls" : 0, 
    "query_score" : 1.00, 
    "gallery_image_id" : [1,2,3,4,5,6,7,8,9,10], 
    "gallery_bbox":[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] 
}

NUM_WORKER = 40

class Retrieval(object):
    def __init__(self, database_folder, save_dir='.'):
        self.database_folder = database_folder
        self.save_dir = save_dir
        self.index_feature = defaultdict(list)
        self.index_id = defaultdict(list)
        self.index_bbox = defaultdict(list)

        self.num_proc = min(NUM_WORKER, os.cpu_count())
        self.retrieval_result = Manager().list()
        self.query_que = Queue()
        self.procs = []

        for _ in range(self.num_proc):
            p = Process(target=self.search)
            self.procs.append(p)

    @staticmethod
    def load_json(json_file):
        with open(json_file, 'r') as fr:
            return json.load(fr)

    def index(self):
        json_files = os.listdir(self.database_folder)
        for json_file in json_files:
            info_dict = self.load_json(os.path.join(self.database_folder, json_file))
            image_name = info_dict["image"].split('.')[0]
            image_id = int(image_name)
            predict_labels = info_dict["labels"]
            predict_bboxes = info_dict["bbox"]
            predict_bboxes, predict_labels = get_filtered(predict_bboxes, predict_labels)
            for item in range(len(predict_labels)):
                predict_category = predict_labels[item]
                feature = np.load(os.path.join(FEAT_FOLDER, "{}-{}.jpg.npy".format(image_name, item))).tolist()
                bbox = [int(_) for _ in predict_bboxes[item]]
                self.index_feature[predict_category].append(feature)
                self.index_bbox[predict_category].append(bbox)
                self.index_id[predict_category].append(image_id)
        print("Index info:")
        total = 0
        for c in range(1, CLASS_NUM):
            print("#items in category-{}:".format(c), len(self.index_id[c]))
            total += len(self.index_id[c])
        print("#items in database:", total)

    def __search_in_category(self, query_category, query_feature):
        if len(self.index_id[query_category]) == 0:
            return TEMPLATE["gallery_image_id"], TEMPLATE["gallery_bbox"]
        query_feature = np.array(query_feature)
        database_feature = np.array(self.index_feature[query_category])
        database_ids = self.index_id[query_category]
        database_bbox = self.index_bbox[query_category]
        similarity = query_feature.dot(database_feature.T)
        sorted_ids = np.argsort(-similarity)[:10]
        sorted_ids = sorted_ids.tolist()
        if RERANK:
            new_query = np.mean(np.concatenate((query_feature.reshape(1,-1), database_feature[sorted_ids[:RERANK]]), axis=0), axis=0)
            similarity = new_query.dot(database_feature.T)
            sorted_ids = np.argsort(-similarity)[:10]
            sorted_ids = sorted_ids.tolist()
        if len(sorted_ids) < 10:
            sorted_ids.extend([sorted_ids[0]] * (10 - len(sorted_ids)))
        gallery_image_id = [database_ids[i] for i in sorted_ids]
        gallery_bbox = [database_bbox[i] for i in sorted_ids]
        return gallery_image_id, gallery_bbox

    def search(self):
        while not self.query_que.empty():
            query_json = self.query_que.get()
            try:
                query_dict = self.load_json(query_json)
            except:
                print("can not open {}".format(json_file))
                continue
            query_name = query_dict["image"].split('.')[0]
            query_image_id = int(query_name)
            print("querying {:06d}".format(query_image_id))
            if len(query_dict["bbox"]) == 0:
                q_result = deepcopy(TEMPLATE)
                q_result["query_image_id"] = query_image_id
                self.retrieval_result.append(q_result)
            else:
                query_bboxes = query_dict["bbox"]
                query_labels = query_dict["labels"]
                query_scores = query_dict["scores"]
                query_bboxes, query_labels, query_scores = get_filtered(query_bboxes, query_labels, query_scores)
                for qitem in range(len(query_labels)):
                    query_feature = np.load(os.path.join(FEAT_FOLDER, "{}-{}.jpg.npy".format(query_name, qitem))).tolist()
                    gallery_image_id, gallery_bbox = self.__search_in_category(query_labels[qitem], query_feature)
                    q_result = {
                        "query_image_id": query_image_id,
                        "query_bbox": [int(_) for _ in query_bboxes[qitem]],
                        "query_cls": query_labels[qitem],
                        "query_score": round(query_scores[qitem], 2),
                        "gallery_image_id": gallery_image_id,
                        "gallery_bbox": gallery_bbox
                    }
                    self.retrieval_result.append(q_result)

    def launch_retrieval(self, query_folder):
        query_jsons = os.listdir(query_folder)
        for query_json in query_jsons:
            self.query_que.put(os.path.join(query_folder, query_json))
        for p in self.procs:
            p.start()
        for p in self.procs:
            p.join()

    def save_result(self):
        retrieval_result = list(self.retrieval_result)
        print("#query_item_num:", len(retrieval_result))
        with open(os.path.join(self.save_dir, 'submission.json'), 'w') as fw:
            json.dump(retrieval_result, fw)


if __name__ == "__main__":
    st = time.time()
    print("Env Folder:", ENV_FOLDER)
    deepfashion = Retrieval(DATABASE_FOLDER, SAVE_DIR)
    deepfashion.index()
    deepfashion.launch_retrieval(QUERY_FOLDER)
    deepfashion.save_result()
    et = time.time()
    print("{} minutes elapsed.".format(round((et-st)/60, 2)))