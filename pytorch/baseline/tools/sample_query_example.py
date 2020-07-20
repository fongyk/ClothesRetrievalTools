import json
import random
import os
import shutil

SAMPLE_NUM = 10
RETRIEVAL_RESULT = "submission.json"
IMAGE_FOLDER = "../bbox_result"
def sample_query():
    with open(RETRIEVAL_RESULT, "r") as fr:
        res = json.load(fr)
        for _ in range(SAMPLE_NUM):
            q = random.randint(0, len(res)-1)
            query_id = res[q]["query_image_id"]
            query_folder = "{:06d}".format(query_id)
            if os.path.exists(query_folder): continue
            os.makedirs(query_folder)
            print("query {}".format(query_folder))
            shutil.copy(os.path.join(IMAGE_FOLDER, query_folder+".jpg"), os.path.join(query_folder, "0-"+query_folder+".jpg"))
            for r, img_id in enumerate(res[q]["gallery_image_id"], 1):
                src_path = os.path.join(IMAGE_FOLDER, "{:06d}.jpg".format(img_id))
                dst_path = os.path.join(query_folder, "{}-{:06d}.jpg".format(r, img_id))
                shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    sample_query()
