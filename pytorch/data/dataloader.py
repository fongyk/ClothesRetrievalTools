from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import os
from PIL import Image

from .dataset import ListDataset, CustomImageFolder

IMG_TRANSFORMS = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_loader(dataset, batch_size, is_distributed=False):
    assert os.path.exists(dataset), "{} doesn't exist.".format(dataset)
    ## dataset-list
    if os.path.isfile(dataset):
        with open(dataset, 'r') as fr:
            img_list = fr.readlines()
            data = ListDataset(img_list, IMG_TRANSFORMS)
            if is_distributed:
                sampler = DistributedSampler(data)
            else:
                sampler = None
            loader = DataLoader(
                dataset=data, 
                shuffle=(sampler is None), 
                num_workers=8, 
                batch_size=batch_size,
                sampler=sampler
            )

    ## dataset-folder
    else:
        data = CustomImageFolder(dataset, transform=IMG_TRANSFORMS)
        if is_distributed:
            sampler = DistributedSampler(data)
        else:
            sampler = None
        loader = DataLoader(
            dataset=data, 
            shuffle=(sampler is None), 
            num_workers=8, 
            batch_size=batch_size,
            sampler=sampler
        )
  
    return loader
