# ClothesRetrievalTools

This is a distributed version by using multiple GPUs in one node.  

## Description

Train a feature embedding network with cross entropy loss and triplet loss.

Input image is cropped based on bounding boxes. Maskrcnn is trained to detect clothes object.

Query images are from users (consumers), gallery images are from shops.

Training sample: a triplet consists of shop/user anchors and shop/user negatives.

## Requirements

- python                    3.6.10
- tensorflow-gpu            2.1.0
- tensorflow-addons         0.9.1
- pydot                     1.4.1
- graphviz                  2.40.1

`pydot` and `graphviz` are expected when calling `plot_model`.

## Specify GPUs

Globally set `os.environ["CUDA_VISIBLE_DEVICES"]`. By default, strategy occupies all available GPUs.

## Trainer and extractor

`trainer`/`fitter`: training with cross-entropy loss and triplet loss.

`extractor`/`predicter`: loading the trained model to extract image features.

At the end of test_data, `extractor` raises
```
RuntimeError: Can't copy Tensor with type string to device /job:localhost/replica:0/task:0/device:GPU:0.
```
Accoring to https://github.com/tensorflow/tensorflow/issues/38343#issuecomment-610853936, it seems to be fixed with TF v2.2.0-rc2.

## Retrieval

- **retrieval_in_category** only search in the same category as the query.

## Note

Run in `tmux` environment could supress progress bar.