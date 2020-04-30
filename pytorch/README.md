# ClothesRetrievalTools

This is a distributed version by using multiple GPUs in one node. 

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
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)(optional)        0.1

## Specify GPUs

Globally set `os.environ["CUDA_VISIBLE_DEVICES"]`. By default, strategy occupies all available GPUs.

## Retrieval

- **retrieval_in_category** only search in the same category.

- **retrieval_all** search in the whole database.
