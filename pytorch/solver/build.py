import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

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
    scheduler = MultiStepLR(optimizer,
                            milestones=args.lr_milestones,
                            gamma=args.lr_gamma
                            )
    return scheduler
