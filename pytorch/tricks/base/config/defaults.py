# Number of image categories for classification task
NUM_CLASSES = 14
# Size of image during training/testing
INPUT_SIZE = [256, 128]
# Value of padding size
INPUT_PADDING = 10
# Image normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Name of backbone
MODEL_NAME = 'resnet50'
# Last stride of backbone
MODEL_LAST_STRIDE = 1
# Path to pretrained model of backbone
MODEL_PRETRAIN_PATH = '/home/fong/.torch/models/resnet50-19c8e357.pth'
# Use pretrained model to initialize backbone: 'imagenet' or 'self'
MODEL_PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
MODEL_NECK = 'bnneck'
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
MODEL_TEST_NECK_FEAT = 'before'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
MODEL_IF_WITH_CENTER = 'no'
# The loss type of metric loss
# If train with label smooth, options: 'on', 'off'
MODEL_IF_LABELSMOOTH = 'off'