from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image

from .dataset import ListDataset, CustomImageFolder

IMG_TRANSFORMS = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_loader(dataset, batch_size):
    assert os.path.exists(dataset), "{} doesn't exist.".format(dataset)
    ## dataset-list
    if os.path.isfile(dataset):
        with open(dataset, 'r') as fr:
            img_list = fr.readlines()
            data = ListDataset(img_list, IMG_TRANSFORMS)
            loader = DataLoader(dataset=data, shuffle=False, num_workers=8, batch_size=batch_size)

    ## dataset-folder
    else:
        data = CustomImageFolder(dataset, transform=IMG_TRANSFORMS)
        loader = DataLoader(dataset=data, shuffle=False, num_workers=8, batch_size=batch_size)
  
    return loader
