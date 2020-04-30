## 安装

```
conda install tensorflow-gpu==2.1
```

```
>>> import tensorflow as tf
>>> tf.config.experimental.list_physical_devices("GPU")
```

阻止显示 tensorflow 的 log/warning：
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
```


## label 表示

`tf.keras.losses.CategoricalCrossentropy`：label 采用 one-hot 编码

`tf.keras.losses.SparseCategoricalCrossentropy`：label 直接用整数

## tf.function

使用`tf.function`修饰，Autograph会优化计算图。

## multi-input, multi-output, multi-loss

https://www.tensorflow.org/guide/keras/train_and_evaluate#other_input_formats_supported

https://zhuanlan.zhihu.com/p/58825710

https://stackoverflow.com/questions/58708074/in-tensorflow-keras-2-0-when-a-model-has-multiple-outputs-how-to-define-a-flex

https://stackoverflow.com/questions/59690188/how-do-i-make-a-multi-output-tensorflow-2-0-neural-network-with-two-different-va

https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py


## triplet loss

https://www.tensorflow.org/addons/api_docs/python/tfa/losses/TripletSemiHardLoss

https://www.tensorflow.org/addons/tutorials/losses_triplet#build_the_model

http://daniel-at-world.blogspot.com/2019/07/implementing-triplet-loss-function-in.html

https://gist.github.com/dkohlsdorf/90df4721cd70c6f4420b7e049796280b

https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py

https://blog.csdn.net/MrCharles/article/details/103284976

https://blog.csdn.net/hustqb/article/details/80361171

## 训练模式与测试模式，batch_norm的行为

https://zhuanlan.zhihu.com/p/64310188

https://pgaleone.eu/tensorflow/keras/2019/01/19/keras-not-yet-interface-to-tensorflow/

## 模型保存与读取

https://tensorflow.google.cn/tutorials/keras/save_and_load

https://tensorflow.google.cn/api_docs/python/tf/train/Checkpoint?hl=en

解决 `unknown loss function` 错误：
https://stackoverflow.com/questions/57982158/valueerror-unknown-loss-functionfocal-loss-fixed-when-loading-model-with-my-cu
```
model = keras.models.load_model("lc_model.h5", custom_objects={'my_loss': my_loss()})

model = keras.models.load_model("lc_model.h5", compile=False)
```

## 教程

https://github.com/czy36mengfei/tensorflow2_tutorials_chinese

https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book

## pretrained, fine-tune

https://www.tensorflow.org/tutorials/images/transfer_learning

## 定义数据集

https://www.tensorflow.org/tutorials/load_data/images#%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA_tfdatadataset

https://www.tensorflow.org/guide/data_performance#prefetching

https://blog.csdn.net/abc13526222160/article/details/100063168

https://blog.csdn.net/sq_damowang/article/details/103291640


## 分布式与多GPU

```
tf.distribute.Strategy
```

https://tensorflow.google.cn/tutorials/distribute/custom_training

https://tensorflow.google.cn/tutorials/distribute/keras

https://www.tensorflow.org/tutorials/distribute/keras

https://www.tensorflow.org/guide/gpu#using_multiple_gpus

https://zhuanlan.zhihu.com/p/88165283

## 定义 callbacks

https://www.tensorflow.org/guide/keras/custom_callback?hl=bg

## 混合精度

https://zhuanlan.zhihu.com/p/103685761

https://www.tensorflow.org/guide/keras/mixed_precision

## weight decay / l2 regularization

https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW


## estimator

https://www.tensorflow.org/tutorials/estimator/premade

## Tensorflow 默认占用GPU所有显存

https://stackoverflow.com/questions/37775979/tensorflow-uses-same-amount-of-gpu-memory-regardless-of-batch-size