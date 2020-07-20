# ClothesRetrievalTools

Using tricks provided by [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

    - random erasing (database augmentation)
    - warmup training
    - label smoothing

## Description

Train a feature embedding network with cross entropy loss and triplet loss.

Input image is cropped based on bounding boxes. Maskrcnn is trained to detect clothes object.

Query images are from users (consumers), gallery images are from shops.

Training sample: a quadruplet consists of shop/user anchors and shop/user negatives.

## Requirements

- python                    3.6.10
- pytorch                   1.1.0
- apex                      0.1
- torchvision               0.2.2
- tensorboardx              1.8
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)(optional)        0.1

## Retrieval

- **retrieval_in_category** only search in the same category.

- **retrieval_all** search in the whole database.
