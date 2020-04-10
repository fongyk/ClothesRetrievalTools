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
