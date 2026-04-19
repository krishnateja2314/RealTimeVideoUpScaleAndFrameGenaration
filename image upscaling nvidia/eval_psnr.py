import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

from dataset import VimeoDataset
from model import ESPCN

# Silence the PyTorch storage warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# ⚠️ PATHS: ADD YOUR SPECIFIC DIRECTORIES HERE ⚠️
# ==============================================================================
# Make sure this points to your TESTING set, not the training set!
TEST_DATA_ROOT = r"D:\vimeo_super_resolution_test" 
LOAD_MODEL_PATH = "espcn_vimeo_deep_final.pth"
# ==============================================================================

def calculate_psnr():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load the Testing Dataset
    test_dataset = VimeoDataset(root_dir=TEST_DATA_ROOT)
    
    # We can use a larger batch size here because inference takes less VRAM than training
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 2. Load the Trained Model
    model = ESPCN(scale_factor=4).to(device)
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device, weights_only=True))
    model.eval() # Locks the weights so they don't change during testing

    # PSNR mathematically requires Mean Squared Error (MSE), not L1 Loss
    criterion = nn.MSELoss() 
    total_mse = 0.0
    
    print(f"Calculating PSNR across {len(test_dataset)} test images...")

    # 3. Run Inference
    with torch.no_grad(): # Tells PyTorch not to calculate gradients, saving massive VRAM
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            outputs = model(lr_imgs)
            
            # Calculate MSE for this batch and add it to our running total
            mse = criterion(outputs, hr_imgs).item()
            total_mse += mse

    # 4. Calculate Final Score
    avg_mse = total_mse / len(test_loader)
    
    if avg_mse == 0:
        print("Perfect match! PSNR is infinite.")
    else:
        # Calculate PSNR. MAX_I is 1.0 because PyTorch tensors are between 0.0 and 1.0
        psnr = 10 * math.log10(1.0 / avg_mse)
        
        print(f"\n==============================")
        print(f"🏆 FINAL EVALUATION RESULTS 🏆")
        print(f"==============================")
        print(f"Average MSE:  {avg_mse:.6f}")
        print(f"Average PSNR: {psnr:.2f} dB")
        print(f"==============================\n")

if __name__ == "__main__":
    calculate_psnr()