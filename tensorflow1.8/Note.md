# 教程

https://github.com/tensorflow/models/tree/r1.5/research/slim

https://github.com/tensorflow/docs/blob/r1.8/site/en/api_docs/python/index.md

mnist_slim.py：https://gist.github.com/terasakisatoshi/49fa3366b428d732c0792255ae49a2a5

TensorFlow-Slim API 官方教程: https://blog.csdn.net/wanttifa/article/details/90208398

中文社区：http://www.tensorfly.cn/tfdoc/api_docs/SOURCE/tutorials/deep_cnn.html

TensorFlow-slim 训练 CNN 分类模型：https://www.jianshu.com/p/68598e8fca7d

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

# 有待改进之处

1. 需要探索更可靠的近邻连接关系。本文构建图连接都是基于图像的原始特征计算欧氏距离，在距离最近的图像之间建立连接。一方面，仅仅靠欧氏距离度量出来的近邻连接并不可靠，甚至存在一些错误的连接，这些连接会严重影响特征增强（CFA）或重表征（GFR）的性能，降低新表征的区分力。另一方面，特征被重复利用：既用来建立连接，又被作为节点的表征，因而特征本身的误差会被放大。除了对欧氏距离施加更多约束（如互为k近邻），在有条件的情况下，最好能够利用特征之外的信息（如文本标签）来协助建立图连接关系。

2. 减少参数量。在多级特征增强方法（CFA）中，有比较多的超参数需要确定，如尺度、近邻个数、聚类粒度等，想要找到全局最优的参数组合是非常困难的。可以探索自适应确定参数的办法，比如根据图像的分布来自适应地确定图连接中近邻的数量以及聚类中心的数量。

3. 探索更复杂的图神经网络在检索任务中的作用。本文提出的基于图卷积神经网络的重表征方法（GFR）采用的是比较浅层的网络，聚合函数的形式也相对简单，在后续的研究过程中，需要探索更深层的网络以及更有效的聚合方法。

4. 改进算法对大数据集的适应性。为了应对规模大的数据集，CFA进行了多粒度聚类，GFR采用局部连接图代替全局连接图。直观上，切分全局连接图进行计算能够显著降低算法复杂度，然而完整的全局连接图能够保证信息传播顺畅而不被打断。因此，如何在不增加时间和空间复杂度的前提下保持较高的检索性能，是值得研究的方向。