# ClothesRetrievalTools

This is a distributed version by using multiple GPUs in one node.  

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

`trainer`: training with [SoftTriple loss](https://arxiv.org/pdf/1909.05235.pdf).

`multi_trainer`: training with cross-entropy loss and SoftTriple loss.

`extractor`: loading the trained model to extract image features.

The pretrained `resnet_v1_50.ckpt` is downloaded from [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained).

## Retrieval

- **retrieval_in_category** only search in the same category as the query.

## Note

Too many ``fetches`` in ``sess.run`` will slow down training. And according to [link-1](https://stackoverflow.com/questions/43844510/is-session-runfetches-guaranteed-to-execute-its-fetches-arguments-in-order) and [link-2](https://github.com/tensorflow/tensorflow/issues/10860), the execution order of nodes in fetch list is undefined.
