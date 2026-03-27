import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()

class VimeoDataset(Dataset):
    def __init__(self, root, list_file):
        with open(list_file, 'r') as f:
            self.samples = f.read().splitlines()
        self.root = root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        im1 = Image.open(f"{self.root}/{path}/im1.png").convert("RGB")
        im2 = Image.open(f"{self.root}/{path}/im2.png").convert("RGB")
        im3 = Image.open(f"{self.root}/{path}/im3.png").convert("RGB")

        im1 = to_tensor(im1)
        im2 = to_tensor(im2)
        im3 = to_tensor(im3)

        return im1, im3, im2