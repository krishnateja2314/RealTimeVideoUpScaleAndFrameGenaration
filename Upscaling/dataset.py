import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VimeoDataset(Dataset):
    def __init__(self, root_dir, lr_crop_size=64, upscale_factor=4):
        """
        root_dir: The main folder containing 'low_resolution', 'target', etc.
        """
        self.root_dir = root_dir
        self.lr_crop_size = lr_crop_size
        self.upscale_factor = upscale_factor
        
        self.target_dir = os.path.join(root_dir, 'target')
        
        # Grab all the .png files from the target directory
        self.hr_paths = glob.glob(os.path.join(self.target_dir, '**', '*.png'), recursive=True)
        
        if len(self.hr_paths) == 0:
            print("Warning: No target images found. Check your dataset path!")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        
        # Clever trick: Replace 'target' in the string with 'low_resolution' to find the matching LR frame
        lr_path = hr_path.replace('target', 'low_resolution')
        
        # Load images
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # --- SYNCHRONIZED RANDOM CROP ---
        lr_w, lr_h = lr_img.size
        
        # Pick a random top-left starting point for the LR crop
        x = random.randint(0, lr_w - self.lr_crop_size)
        y = random.randint(0, lr_h - self.lr_crop_size)
        
        # Crop LR image
        lr_crop = lr_img.crop((x, y, x + self.lr_crop_size, y + self.lr_crop_size))
        
        # Calculate the exact matching coordinates for the High-Res target
        hr_x = x * self.upscale_factor
        hr_y = y * self.upscale_factor
        hr_crop_size = self.lr_crop_size * self.upscale_factor
        
        # Crop HR image
        hr_crop = hr_img.crop((hr_x, hr_y, hr_x + hr_crop_size, hr_y + hr_crop_size))
        
        # Convert to PyTorch Tensors
        to_tensor = transforms.ToTensor()
        
        return to_tensor(lr_crop), to_tensor(hr_crop)

# --- Quick Test Block ---
if __name__ == "__main__":
    # Replace with the path to the folder that HOLDS 'low_resolution' and 'target'
    DATASET_DIR = "D:\\vimeo_super_resolution_test.zip\\vimeo_super_resolution_test" 
    
    dataset = VimeoDataset(root_dir=DATASET_DIR, lr_crop_size=64, upscale_factor=4)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    lr_batch, hr_batch = next(iter(dataloader))
    print(f"Low-Res Input Shape: {lr_batch.shape}")   # Expected: [16, 3, 64, 64]
    print(f"High-Res Target Shape: {hr_batch.shape}") # Expected: [16, 3, 256, 256]