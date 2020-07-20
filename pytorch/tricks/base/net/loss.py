import torch
from torch import nn
import torch.nn.functional as F

class tripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.1):
        super(tripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class tripletLossWithBatchHard(nn.Module):
    """
    Triplet loss with batch hard negative mining.
    """

    def __init__(self, margin=0.1):
        super(tripletLossWithBatchHard, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, label, size_average=True):
        '''
        anchor, positive are embeddings which have been l2-normalized in advance.
        '''
        n, d = anchor.size()
        distance = torch.full((n, n), 2.0, dtype=anchor.dtype, device=anchor.device) - 2.0 * torch.matmul(anchor, positive.t())
        distance_pos = torch.diag(distance)
        ## hard mining
        label = label.view(n, 1)
        mask_neg = (label != label.t())
        distance_by_mask_neg = torch.where(mask_neg > 0, distance, torch.full_like(distance, 4.0))
        distance_row_min = torch.min(distance_by_mask_neg, dim=1, keepdim=True)[0]
        distance_col_min = torch.min(distance_by_mask_neg, dim=0, keepdim=True)[0]
        distance_hard_neg = torch.min(torch.cat((distance_row_min, distance_col_min.t()), dim=1), dim=1)[0]

        losses = F.relu(distance_pos - distance_hard_neg + self.margin)
        return losses.mean() if size_average else losses.sum()

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss