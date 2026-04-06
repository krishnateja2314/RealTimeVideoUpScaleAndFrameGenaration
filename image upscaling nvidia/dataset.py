import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class VimeoDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found at: {list_file}")

        with open(list_file, 'r') as f:
            all_folders = [line.strip() for line in f if line.strip()]
        
        self.samples = []
        print(f"Scanning files in {root_dir}... This may take a moment.")

        for folder in all_folders:
            clean_folder = folder.replace('/', os.sep).replace('\\', os.sep)
            for i in range(1, 8):
                img_name = f"im{i}.png"
                rel_path = os.path.join(clean_folder, img_name)
                hr_path = os.path.join(self.root_dir, 'target', rel_path)
                lr_path = os.path.join(self.root_dir, 'low_resolution', rel_path)
                if os.path.exists(hr_path) and os.path.exists(lr_path):
                    self.samples.append(rel_path)
        
        print(f"✅ Filtered! Found {len(self.samples)} valid frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        lr_path = os.path.join(self.root_dir, 'low_resolution', rel_path)
        hr_path = os.path.join(self.root_dir, 'target', rel_path)

        try:
            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = Image.open(hr_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Corrupt image at {rel_path}, replacing with zeros")
            # Return blank tensors if image is corrupt
            return torch.zeros(3, 64, 64), torch.zeros(3, 256, 256)

        to_tensor = T.ToTensor()
        lr_tensor = to_tensor(lr_img).clamp(0, 1)
        hr_tensor = to_tensor(hr_img).clamp(0, 1)

        # Extra safety - check for NaN in tensors themselves
        if torch.isnan(lr_tensor).any() or torch.isnan(hr_tensor).any():
            print(f"⚠️ NaN in image tensors at {rel_path}, replacing with zeros")
            return torch.zeros(3, 64, 64), torch.zeros(3, 256, 256)

        return lr_tensor, hr_tensor