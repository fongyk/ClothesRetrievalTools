# setting
backbone: resnet50

bn-neck: **before**. "after" mode causes performance degradation.

    - coarse-grained classification (cross entropy loss)
    - fine-grained metric learning (triplet loss)

train num.: 25000, [12000, 20000]

lr: 1e-4

use-amp: True

# baseline 1

loss: softmax + triplet

margin: 0.2

omega: 0.9

training feature normalization: True

accuracy:

    - top-1
      0.24133473378
    - top-5
      0.390886321403
    - top-10
      0.458026985538

# baseline 2

loss: softmax + triplet

margin: 0.3

omega: 0.9

training feature normalization: False

accuracy:

    - top-1
      0.277126928981
    - top-5
      0.425870566373
    - top-10
      0.495111901107

# baseline 3

loss: softmax + triplet

margin: 0.2

omega: 0.7

training feature normalization: True

accuracy:

    - top-1
      0.230346610649
    - top-5
      0.379009453018
    - top-10
      0.449058737982

# baseline 2 + random erasing

loss: softmax + triplet

margin: 0.3

omega: 0.9

training feature normalization: False

accuracy:

    - top-1
      0.30193100105
    - top-5
      0.448089197705
    - top-10
      0.512402036035

# baseline 1 + random erasing

loss: softmax + triplet

margin: 0.3

omega: 0.9

training feature normalization: True

accuracy:

    - top-1
      0.256928173225
    - top-5
      0.397673103337
    - top-10
      0.468934313646

# baseline 2 + random erasing + warmup training

loss: softmax + triplet

margin: 0.3

omega: 0.9

training feature normalization: False

accuracy:

    - top-1
      0.302173386119
    - top-5
      0.452128948857
    - top-10
      0.516441787186

# baseline 2 + random erasing + warmup training + label smooth

loss: softmax + triplet

margin: 0.3

omega: 0.9

training feature normalization: False

accuracy:

    - top-1
      0.307505857639
    - top-5
      0.457461420377
    - top-10
      0.52322856912

# baseline 2 + random erasing + warmup training + label smooth

loss: softmax + triplet

margin: 0.3

omega: 0.7

training feature normalization: False

accuracy:

    - top-1
      0.305809162156
    - top-5
      0.45035145835
    - top-10
      0.512725216127

# se_resnet50 + random erasing + warmup training + label smooth

loss: softmax + triplet

margin: 0.3

omega: 0.7

training feature normalization: False

accuracy:

    - top-1
      0.30459723681
    - top-5
      0.454310414478
    - top-10
      0.5189464329
