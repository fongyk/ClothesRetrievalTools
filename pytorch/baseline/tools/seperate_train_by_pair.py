import os
import json
import shutil
import cv2

train_dir = '/data6/fong/DeepFashion/unzip/train/'
seperate_dir = '/data6/fong/DeepFashion/train_seperate_crop_with_label_to_category/'

def seperate_images(train_dir, seperate_dir, crop=False, to_category=False):
    """
    crop: crop image region.
    to_category: seperate images by pair in each category.
    """
    annos = os.listdir(os.path.join(train_dir, 'annos'))
    for anno in annos:
        prefix = anno.split('.')[0]
        anno = os.path.join(train_dir, 'annos', anno)
        with open(anno, 'r') as fr:
            info = json.load(fr)
            for attr in list(info.keys()):
                if attr.startswith("item"):
                    category = info[attr]['category_id']
                    src = os.path.join(train_dir, 'image', prefix+'.jpg')
                    if to_category:
                        dst = os.path.join(seperate_dir, str(category), 'pair_id_'+str(info['pair_id']), 'style_'+str(info[attr]['style']), info['source'])
                    else:
                        dst = os.path.join(seperate_dir, 'pair_id_'+str(info['pair_id']), 'style_'+str(info[attr]['style']), info['source'])
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    if not crop:
                        try:
                            shutil.copy(src, dst)
                            print(src)
                        except Exception as e:
                            print(e)
                    else:
                        try:
                            image = cv2.imread(src)
                            x1, y1, x2, y2 = info[attr]['bounding_box']
                            region = image[y1:y2, x1:x2]
                            cv2.imwrite(os.path.join(dst, prefix+'.{}.jpg'.format(category)), region)
                            print(src)
                        except Exception as e:
                            print(e)

if __name__ == "__main__":
    seperate_images(train_dir, seperate_dir, crop=True, to_category=True)