# ClothesRetrievalTools

## Description

Train a feature embedding network with cross entropy loss and triplet loss.

Input image is cropped based on bounding boxes. Maskrcnn is trained to detect clothes object.

Query images are from users (consumers), gallery images are from shops.

Training sample: a triplet consists of shop/user anchors and shop/user negatives.

## Requirements

- python                    3.6.10
- tensorflow-gpu            1.8.0
- numpy                     1.16.0
- cuda                      9.0

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
export PATH=$PATH:/usr/local/cuda-9.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-9.0
```

## Trainer and extractor

`trainer`: training with cross-entropy loss and triplet loss.

`extractor`: loading the trained model to extract image features.

The pretrained `resnet_v1_50.ckpt` is downloaded from [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained).

## Retrieval

- **retrieval_in_category** only search in the same category as the query.