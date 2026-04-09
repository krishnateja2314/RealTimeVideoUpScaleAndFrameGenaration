import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=48):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size

        self.hr_images = sorted(os.listdir(hr_dir))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):

        hr_name = self.hr_images[idx]
        lr_name = hr_name.replace(".png", "x4.png")

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(self.lr_dir, lr_name)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # ---- Random Patch Cropping ----
        lr_w, lr_h = lr.size
        x = random.randint(0, lr_w - self.patch_size)
        y = random.randint(0, lr_h - self.patch_size)

        lr_patch = lr.crop((x, y, x+self.patch_size, y+self.patch_size))

        scale = 4
        hr_patch = hr.crop((
            x*scale,
            y*scale,
            (x+self.patch_size)*scale,
            (y+self.patch_size)*scale
        ))

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)