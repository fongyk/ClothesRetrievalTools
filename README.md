# ClothesRetrievalTools

This is a distributed version by using multiple GPUs in one node. 

## Data organization

```
category
└── pair_id
    └── style
        ├── shop
        │   ├── xxxx.jpg
        │   └── xxxx.jpg
        └── user
            ├── xxxx.jpg
            └── xxxx.jpg
```

Query images are from users (consumers), gallery images are from shops.

Images iif with same pair_id and style are positive pairs.

## Reference

- https://sites.google.com/view/cvcreative2020/deepfashion2

- https://competitions.codalab.org/competitions/22967