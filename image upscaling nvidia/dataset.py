import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class VimeoDataset(Dataset):
    def __init__(self, root_dir, list_file):
        self.root_dir = root_dir
        
        with open(list_file, 'r', encoding='utf-8') as f:
            all_folders = [line.strip() for line in f if line.strip()]
        
        self.samples = []
        print(f"Scanning {root_dir} for 7-frame sequences...")

        for folder in all_folders:
            clean_folder = folder.replace('/', os.sep).replace('\\', os.sep)
            
            # Check if all 7 low-res frames and the target high-res im4 exist
            valid = True
            for i in range(1, 8):
                lr_path = os.path.join(self.root_dir, 'low_resolution', clean_folder, f"im{i}.png")
                if not os.path.exists(lr_path):
                    valid = False
                    break
            hr_path = os.path.join(self.root_dir, 'target', clean_folder, "im4.png")
            if not os.path.exists(hr_path):
                valid = False
            
            if valid:
                self.samples.append(clean_folder)
        
        print(f"✅ VSR Dataset ready! Found {len(self.samples)} valid sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        to_tensor = T.ToTensor()
        
        # Load all 7 low-res frames
        lr_frames = []
        for i in range(1, 8):
            lr_path = os.path.join(self.root_dir, 'low_resolution', folder, f"im{i}.png")
            img = Image.open(lr_path).convert('RGB')
            lr_frames.append(to_tensor(img))
            
        # Stack them into a single tensor with 21 channels (7 frames * 3 colors)
        lr_tensor = torch.cat(lr_frames, dim=0)

        # Load ONLY the target hr im4.png
        hr_path = os.path.join(self.root_dir, 'target', folder, "im4.png")
        hr_img = Image.open(hr_path).convert('RGB')
        hr_tensor = to_tensor(hr_img)

        return lr_tensor, hr_tensor