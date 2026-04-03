import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class VimeoDataset(Dataset):
    def __init__(self, root, list_file, crop_size=(256, 256)):
        with open(list_file, 'r') as f:
            self.samples = f.read().splitlines()
        self.root = root
        self.crop_size = crop_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        im1 = Image.open(f"{self.root}/{path}/im1.png").convert("RGB")
        im2 = Image.open(f"{self.root}/{path}/im2.png").convert("RGB")
        im3 = Image.open(f"{self.root}/{path}/im3.png").convert("RGB")

        # Random Crop
        w, h = im1.size
        th, tw = self.crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        im1 = im1.crop((j, i, j + tw, i + th))
        im2 = im2.crop((j, i, j + tw, i + th))
        im3 = im3.crop((j, i, j + tw, i + th))

        # Random Horizontal Flip
        if random.random() > 0.5:
            im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
            im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
            im3 = im3.transpose(Image.FLIP_LEFT_RIGHT)

        to_tensor = transforms.ToTensor()
        return to_tensor(im1), to_tensor(im3), to_tensor(im2)