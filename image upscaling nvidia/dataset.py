import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=48, scale=4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale

        # Get HR filenames
        self.hr_images = sorted(os.listdir(hr_dir))

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):

        # -----------------------------
        # File names
        # -----------------------------
        hr_name = self.hr_images[idx]
        lr_name = hr_name.replace(".png", f"x{self.scale}.png")

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(self.lr_dir, lr_name)

        # -----------------------------
        # Load images
        # -----------------------------
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        hr = np.array(hr)
        lr = np.array(lr)

        # -----------------------------
        # Random Crop (VERY IMPORTANT)
        # -----------------------------
        h, w, _ = lr.shape

        ps = self.patch_size

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        lr_patch = lr[y:y+ps, x:x+ps]
        hr_patch = hr[
            y*self.scale:(y+ps)*self.scale,
            x*self.scale:(x+ps)*self.scale
        ]

        # -----------------------------
        # Convert to tensor
        # -----------------------------
        lr_patch = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0
        hr_patch = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0

        return lr_patch, hr_patch