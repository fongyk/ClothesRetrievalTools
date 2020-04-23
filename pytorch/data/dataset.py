from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import sys
from PIL import Image


class ListDataset(Dataset):
    """
    load image from list
    """
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform
        self.img_num = len(img_list)

    def __getitem__(self, index):
        anchor = self.img_list[index].strip('\n')
        img_name = anchor.split('/')[-1]
        try:
            anchor_img = Image.open(anchor)
        except OSError:
            print("can not open {}.".format(anchor))
            sys.exit(1)

        if anchor_img.mode != "RGB":
            origin = anchor_img
            anchor_img = Image.new("RGB", origin.size)
            anchor_img.paste(origin)
        
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)

        return anchor_img, img_name

    def __len__(self):
        return self.img_num


class CustomImageFolder(ImageFolder):
    """
    ImageFloader return (img, filename)

    default folder tree:
    dir
    .
    ├──dog
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──cat  
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──...
    """
    def __init__(self, dir, transform=None):
        super(CustomImageFolder, self).__init__(dir, transform)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        filename = path.split('/')[-1]
        # return super(CustomImageFolder, self).__getitem__(index), filename
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename


    def __len__(self):
        return len(self.imgs)