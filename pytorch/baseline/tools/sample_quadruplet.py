import os
import glob
import random

num_pairs = 200000
seperate_path = '/data6/fong/DeepFashion/train_seperate_crop_with_label_to_category'
quadruple_path = '/data6/fong/DeepFashion/quadruple_crop_with_label_to_category'

def sample_quadruple_random(seperate_path, num_pairs, verbose=False):
    pair_ids = os.listdir(seperate_path)
    pair_ids.sort()
    n = 0
    finished = False
    while not finished:
        for pid in pair_ids:
            ## sample shop_anchor and user_anchor
            styles = os.listdir(os.path.join(seperate_path, pid))
            if 'style_1' not in styles: continue
            while True:
                style_anchor = random.choice(styles)
                if style_anchor != 'style_0':
                    break
            shop_user = os.listdir(os.path.join(seperate_path, pid, style_anchor))
            if len(shop_user) < 2: continue
            n += 1
            if n > num_pairs:
                finished = True
                break
            print(n, pid)
            shop_anchor = random.choice(os.listdir(os.path.join(seperate_path, pid, style_anchor, 'shop')))
            user_anchor = random.choice(os.listdir(os.path.join(seperate_path, pid, style_anchor, 'user')))
            ## sample shop_negative and user_negative in one pair and one style
            ok = True
            while ok:
                pid_negative = random.choice(pair_ids)
                if pid_negative != pid:
                    negative_styles = os.listdir(os.path.join(seperate_path, pid_negative))
                    if len(negative_styles) > 1:
                        while True:
                            style_negative = random.choice(negative_styles)
                            if style_negative != 'style_0':
                                shop_user = os.listdir(os.path.join(seperate_path, pid_negative, style_negative))
                                if len(shop_user) < 2: break
                                shop_negative = random.choice(os.listdir(os.path.join(seperate_path, pid_negative, style_negative, 'shop')))
                                user_negative = random.choice(os.listdir(os.path.join(seperate_path, pid_negative, style_negative, 'user')))
                                ok = False
                                break
            if verbose:
                shop_anchor = os.path.join(seperate_path, pid, style_anchor, 'shop', shop_anchor)
                user_anchor = os.path.join(seperate_path, pid, style_anchor, 'user', user_anchor)
                shop_negative = os.path.join(seperate_path, pid_negative, style_negative, 'shop', shop_negative)
                user_negative = os.path.join(seperate_path, pid_negative, style_negative, 'user', user_negative)
            quadruple = shop_anchor + ' ' + user_anchor + ' ' + shop_negative + ' ' + user_negative + '\n'
            quadr_file = os.path.join(quadruple_path, "{:06d}".format(n))
            with open(quadr_file, 'w') as fw:
                fw.write(quadruple)

def sample_quadruple_with_category(seperate_path, num_pairs, verbose=True):
    category_ids = os.listdir(seperate_path)
    category_ids.sort()
    n = 0
    finished = False
    while not finished:
        for cid in category_ids:
            pair_ids = os.listdir(os.path.join(seperate_path, cid))
            if len(pair_ids) < 2: continue
            for pid in pair_ids:
                ## sample shop_anchor and user_anchor
                styles = os.listdir(os.path.join(seperate_path, cid, pid))
                if 'style_1' not in styles: continue
                while True:
                    style_anchor = random.choice(styles)
                    if style_anchor != 'style_0':
                        break
                shop_user = os.listdir(os.path.join(seperate_path, cid, pid, style_anchor))
                if len(shop_user) < 2: continue
                shop_anchor = random.choice(glob.glob(os.path.join(seperate_path, cid, pid, style_anchor, 'shop', '*.jpg')))
                user_anchor = random.choice(glob.glob(os.path.join(seperate_path, cid, pid, style_anchor, 'user', '*.jpg')))
                ## sample shop_negative and user_negative in one pair, and they are in same category with anchor.
                while True:
                    pid_negative = random.choice(pair_ids)
                    if pid_negative != pid:
                        break
                shop_negatives = glob.glob(os.path.join(seperate_path, cid, pid_negative, 'style_[1-9]', 'shop', '*.jpg'))
                user_negatives = glob.glob(os.path.join(seperate_path, cid, pid_negative, 'style_[1-9]', 'user', '*.jpg'))
                if len(shop_negatives) < 1 or len(user_negatives) < 1: continue
                shop_negative = random.choice(shop_negatives)
                user_negative = random.choice(user_negatives)
                if not verbose:
                    shop_anchor = shop_anchor.split('/')[-1]
                    user_anchor = user_anchor.split('/')[-1]
                    shop_negative = shop_negative.split('/')[-1]
                    user_negative = user_negative.split('/')[-1]
                quadruple = shop_anchor + ' ' + user_anchor + ' ' + shop_negative + ' ' + user_negative + '\n'
                quadr_file = os.path.join(quadruple_path, "{:06d}".format(n))
                with open(quadr_file, 'w') as fw:
                    fw.write(quadruple)
                n += 1
                if n > num_pairs: 
                    finished = True
                    break
                print(n, pid)

def check_nth_quadruple(n):
    import json
    ann_path = '/data6/fong/DeepFashion/unzip/train/annos'
    quadr_file = os.path.join(quadruple_path, "{:06d}".format(n))
    with open(quadr_file, 'r') as fr:
        imgs = fr.readline().strip().split(' ')
        for im in imgs:
            print(im)
            ann = os.path.join(ann_path, im.split('.')[0]+'.json')
            print(ann)
            with open(ann, 'r') as ar:
                d = json.load(ar)
                print("pair: {}, source: {}".format(d['pair_id'], d['source']))
                for k in list(d.keys()):
                    if "item" in k:
                        print("style:{}, category: {}".format(d[k]['style'], d[k]['category_id']))


if __name__ == "__main__":
    # sample_quadruple_random(seperate_path, num_pairs, verbose=True)
    # check_nth_quadruple(5122)
    sample_quadruple_with_category(seperate_path, num_pairs, verbose=True)

    
