import torch
from torch import nn
import torch.nn.functional as F

from .resnet import ResNet, BasicBlock, Bottleneck
from .senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .resnet_ibn_a import resnet50_ibn_a
from .loss import tripletLoss, tripletLossWithBatchHard, CrossEntropyLabelSmooth
from ..config import defaults as df

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BaseNet(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=df.NUM_CLASSES, last_stride=df.MODEL_LAST_STRIDE, model_path=df.MODEL_PRETRAIN_PATH, neck=df.MODEL_NECK, neck_feat=df.MODEL_TEST_NECK_FEAT, model_name=df.MODEL_NAME, pretrain_choice=df.MODEL_PRETRAIN_CHOICE, margin=0.1, omega=0.5, use_hardtriplet=False, use_labelsmooth=False):
        super(BaseNet, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        
        self.tl = tripletLoss(margin)
        self.htl = tripletLossWithBatchHard(margin)
        self.xe = CrossEntropyLabelSmooth(num_classes) if use_labelsmooth else nn.CrossEntropyLoss()
        self.omega = omega
        self.use_hardtriplet = use_hardtriplet

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (b, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        cls_score = self.classifier(feat)
        if self.training:
            return global_feat, cls_score  # global feature for triplet loss
            # return F.normalize(global_feat, p=2, dim=1), cls_score 
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return F.normalize(feat, p=2, dim=1), cls_score 
            else:
                # print("Test with feature before BN")
                return F.normalize(global_feat, p=2, dim=1), cls_score 

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    
    def loss(self, *args):
        if self.use_hardtriplet:
            return self.hard_loss(*args)
        else:
            return self.base_loss(*args)

    def base_loss(self, shop_a, 
                   shop_a_label, 
                   user_a, 
                   user_a_label, 
                   shop_n,
                   shop_n_label,
                   user_n,
                   user_n_label
                   ):
        shop_a, shop_a_predict = self.forward(shop_a)
        user_a, user_a_predict = self.forward(user_a)
        shop_n, shop_n_predict = self.forward(shop_n)
        user_n, user_n_predict = self.forward(user_n)

        triplet_loss = (self.tl(shop_a, user_a, shop_n) + self.tl(user_a, shop_a, user_n)) * 0.5
        xentropy_loss = (self.xe(shop_a_predict, shop_a_label) + self.xe(user_a_predict, user_a_label) + self.xe(shop_n_predict, shop_n_label) + self.xe(user_n_predict, user_n_label)) * 0.25

        loss = self.omega * triplet_loss + (1 - self.omega) * xentropy_loss

        return dict(loss=loss, triplet_loss=triplet_loss, xentropy_loss=xentropy_loss)

    def hard_loss(self, shop_a, 
                        shop_a_label, 
                        user_a, 
                        *args
                        ):
        shop_a, shop_a_predict = self.forward(shop_a)
        user_a, user_a_predict = self.forward(user_a)
        hard_triplet_loss = self.htl(shop_a, user_a, shop_a_label)
        xentropy_loss = 0.5 * (self.xe(shop_a_predict, shop_a_label) + self.xe(user_a_predict, shop_a_label))

        loss = self.omega * hard_triplet_loss + (1 - self.omega) * xentropy_loss

        return dict(loss=loss, hard_triplet_loss=hard_triplet_loss, xentropy_loss=xentropy_loss)

