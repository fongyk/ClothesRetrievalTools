# ClothesRetrievalTools

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

## Trainer and extractor

`trainer`: training with cross-entropy loss and triplet loss.

`extractor`: loading the trained model to extract image features.

Two modes are provided: `train-extract` and `fit-predict`。
- when `train` mode (custom training) are adopted in trainer, then use `extract` mode in extractor.
    - `train`: training with pre-sampled triplets.
- when `fit` mode are adopted in trainer, then use `predict` mode in extractor.
    - `fit`: training with batch mining through `tensorflow_addons.losses.TripletSemiHardLoss`.



## Retrieval

- **retrieval_in_category** only search in the same category as the query.