import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler

class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return DistributedSampler(dataset)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
    dataset, sampler, images_per_batch, num_iters=None, start_iter=0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_dataloader(
        dataset, 
        images_per_gpu, 
        num_iters, 
        start_iter=0, 
        shuffle=True,
        is_distributed=False
    ):
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, images_per_gpu, num_iters, start_iter
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_sampler=batch_sampler,
    )
    return dataloader