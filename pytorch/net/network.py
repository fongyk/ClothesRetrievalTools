import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .loss import tripletLoss, tripletLossWithBatchHard

CATEGORY_NUM = 14
## 13 categories + 1

class BaseNet(nn.Module):
    def __init__(self, margin=0.1, omega=0.5, use_hardtriplet=False):
        super(BaseNet, self).__init__()
        self.tl = tripletLoss(margin)
        self.htl = tripletLossWithBatchHard(margin)
        self.xe = nn.CrossEntropyLoss()
        self.omega = omega
        self.use_hardtriplet = use_hardtriplet

    def init_weight(self):
        for module in self.fc.children():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        predict = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x, predict

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
                        _ignore_1, _ignore_2, _ignore_3, _ignore_4, _ignore_5
                        ):
        shop_a, shop_a_predict = self.forward(shop_a)
        user_a, user_a_predict = self.forward(user_a)
        hard_triplet_loss = self.htl(shop_a, user_a, shop_a_label)
        xentropy_loss = 0.5 * (self.xe(shop_a_predict, shop_a_label) + self.xe(user_a_predict, shop_a_label))

        loss = self.omega * hard_triplet_loss + (1 - self.omega) * xentropy_loss

        return dict(loss=loss, hard_triplet_loss=hard_triplet_loss, xentropy_loss=xentropy_loss)



class AlexNet(BaseNet):
    def __init__(self, margin=0.1, omega=0.5, use_hardtriplet=False):
        super(AlexNet, self).__init__(margin, omega, use_hardtriplet)
        alexnet = models.alexnet(pretrained=True)
        self.feature = nn.Sequential(*(alexnet.features[i] for i in range(12)))
        self.feature.add_module('12: Global Pooling', nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, CATEGORY_NUM),
        )
        self.init_weight()

class VGGNet(BaseNet):
    def __init__(self, margin=0.1, omega=0.5, use_hardtriplet=False):
        super(VGGNet, self).__init__(margin, omega, use_hardtriplet)
        vgg = models.vgg19(pretrained=True)
        self.feature = nn.Sequential(*(vgg.features[i] for i in range(36)))
        self.feature.add_module('36: GlobalPooling', nn.AdaptiveAvgPool2d(1))
        # self.batch_norm = nn.BatchNorm1d(512)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, CATEGORY_NUM),
        )
        self.init_weight()

class ResNet(BaseNet):
    def __init__(self, margin=0.1, omega=0.5, use_hardtriplet=False):
        super(ResNet, self).__init__(margin, omega, use_hardtriplet)
        resnet = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(resnet.children())[:-2])
        self.feature.add_module('8: Global Pooling', nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, CATEGORY_NUM),
        )
        self.init_weight()