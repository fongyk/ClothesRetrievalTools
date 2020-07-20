from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image

from .dataset import ListDataset, CustomImageFolder
from .transforms import build_transforms

def get_loader(dataset, batch_size, is_train=False):
    assert os.path.exists(dataset), "{} doesn't exist.".format(dataset)
    img_transforms = build_transforms(is_train)
    ## dataset-list
    if os.path.isfile(dataset):
        with open(dataset, 'r') as fr:
            img_list = fr.readlines()
            data = ListDataset(img_list, img_transforms)
            loader = DataLoader(dataset=data, shuffle=False, num_workers=8, batch_size=batch_size)

    ## dataset-folder
    else:
        data = CustomImageFolder(dataset, transform=img_transforms)
        loader = DataLoader(dataset=data, shuffle=False, num_workers=8, batch_size=batch_size)
  
    return loader
