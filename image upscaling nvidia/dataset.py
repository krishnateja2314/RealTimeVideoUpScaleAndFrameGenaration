import os
import glob
import random 
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF 

class VimeoDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir should be the main folder that contains BOTH 
        'low_resolution' and 'target' subfolders.
        """
        self.lr_dir = os.path.join(root_dir, 'low_resolution')
        self.hr_dir = os.path.join(root_dir, 'target')
        
        # Search for all im4.png files inside the target directory's subfolders
        search_pattern = os.path.join(self.hr_dir, '*', '*', 'im4.png')
        self.hr_image_paths = sorted(glob.glob(search_pattern))
        
        if len(self.hr_image_paths) == 0:
            print(f"Warning: No im4.png files found in {search_pattern}")
            
        self.transform = ToTensor()

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_image_paths[idx]
        lr_path = hr_path.replace('target', 'low_resolution')

        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        # --- DATA AUGMENTATION ---
        # 50% chance to flip both images horizontally
        if random.random() > 0.5:
            hr_image = TF.hflip(hr_image)
            lr_image = TF.hflip(lr_image)
            
        # 50% chance to flip both images vertically
        if random.random() > 0.5:
            hr_image = TF.vflip(hr_image)
            lr_image = TF.vflip(lr_image)
        # ------------------------------

        hr_tensor = self.transform(hr_image)
        lr_tensor = self.transform(lr_image)

        return lr_tensor, hr_tensor