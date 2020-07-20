import torch
from torch import optim
from bisect import bisect_right

def make_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.learning_rate
        weight_decay = args.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = optim.Adam(params, lr)

    return optimizer


def make_scheduler(args, optimizer):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=args.lr_milestones,
                            gamma=args.lr_gamma
                            )
    return scheduler


class WarmupMultiStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def make_warmup_scheduler(args, optimizer):
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
        warmup_factor=args.warmup_factor,
        warmup_iters=args.warmup_iters,
        warmup_method=args.warmup_method
    )

    return scheduler