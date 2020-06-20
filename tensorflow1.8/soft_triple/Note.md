# 教程

https://github.com/tensorflow/models/tree/r1.5/research/slim

https://github.com/tensorflow/docs/blob/r1.8/site/en/api_docs/python/index.md

mnist_slim.py：https://gist.github.com/terasakisatoshi/49fa3366b428d732c0792255ae49a2a5

TensorFlow-Slim API 官方教程: https://blog.csdn.net/wanttifa/article/details/90208398

中文社区：http://www.tensorfly.cn/tfdoc/api_docs/SOURCE/tutorials/deep_cnn.html

TensorFlow-slim 训练 CNN 分类模型：https://www.jianshu.com/p/68598e8fca7d

Inside TF-Slim(12) ：https://zhuanlan.zhihu.com/p/36198988

# 导入预训练模型

TensorFlow 使用预训练模型 ResNet-50：https://www.jianshu.com/p/0237ebbee5d5

使用end_points提取特征：https://stackoverflow.com/questions/49677010/how-to-extract-features-from-the-last-layer-of-slim-tensorflow-inceptionnet-v3-m

https://stackoverflow.com/questions/48947083/re-train-pre-trained-resnet-50-model-with-tf-slim-for-classification-purposes

TF-slim 调用slim提供的网络模型训练自己的数据：https://blog.csdn.net/wc781708249/article/details/78414930

网络微调：
https://github.com/wucng/TensorExpand/blob/master/TensorExpand/%E5%9B%BE%E7%89%87%E9%A1%B9%E7%9B%AE/5%E3%80%81%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0/VGG%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0/%E7%BD%91%E7%BB%9C%E5%BE%AE%E8%B0%83.py

**固定权重+finetune+取出指定层的张量+参数初始化**：https://www.cnblogs.com/yanshw/p/12432595.html
```python
tf.reset_default_graph()        ### 这句暂时可忽略

# 构建计算图
images = tf.placeholder(tf.float32,(None,224,224,3))
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v2.mobilenet(images,depth_multiplier=1.4)

with tf.variable_scope("finetune_layers"):
    mobilenet_tensor = tf.get_default_graph().get_tensor_by_name("MobilenetV2/expanded_conv_14/output:0")       # 获取目标张量，取出mobilenet中指定层的张量

    # 将张量作为新的 Input 向新层传递
    x = tf.layers.Conv2D(filters=256,kernel_size=3,name="conv2d_1")(mobilenet_tensor)
```

# Siamese network， multi-loss

siamese：https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/train.py

Triplet-Loss：https://stackoverflow.com/questions/49099938/triplet-loss-not-converging-using-tensorflow

https://stackoverflow.com/questions/38260113/implementing-contrastive-loss-and-triplet-loss-in-tensorflow

https://github.com/omoindrot/tensorflow-triplet-loss

TensorFlow 训练多任务多标签模型：https://www.jianshu.com/p/270d5de903b6

Multi-Task Learning：https://medium.com/@kajalgupta/multi-task-learning-with-deep-neural-networks-7544f8b7b4e3


# 学习率衰减

学习率衰减/learning rate decay：https://blog.csdn.net/akadiao/article/details/79560731

mnist实例：https://www.cnblogs.com/baby-lily/p/10962574.html

# 模型保存与读取

Loading two models from Saver in the same Tensorflow session：https://stackoverflow.com/questions/41607144/loading-two-models-from-saver-in-the-same-tensorflow-session

模型的保存与恢复(Saver)：https://www.cnblogs.com/denny402/p/6940134.html

# two graphs

https://stackoverflow.com/questions/41607144/loading-two-models-from-saver-in-the-same-tensorflow-session

# 分布式

使用tensorflow1.x实现单机多卡训练：https://zhuanlan.zhihu.com/p/102298061

multigpu_cnn：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_cnn.py

https://github.com/tensorflow/models/blob/r1.5/research/slim/train_image_classifier.py

https://github.com/tensorflow/models/blob/r1.5/tutorials/image/cifar10/cifar10_multi_gpu_train.py

tensorflow 多GPU编程 完全指南：https://blog.csdn.net/minstyrain/article/details/80986397

Batch Normalization for Multi-GPU / Data Parallelism：https://github.com/tensorflow/tensorflow/issues/7439
