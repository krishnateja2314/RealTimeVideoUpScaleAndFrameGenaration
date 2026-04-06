import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from dataset import VimeoDataset
from model import FlowNet
from torchvision.transforms import ToPILImage
import math

# Using the built-in metrics from a common library or manual calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def main():
    device = torch.device('cuda')
    DATA_ROOT = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/sequences"
    TEST_LIST = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/tri_testlist.txt"

    dataset = VimeoDataset(DATA_ROOT, TEST_LIST)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = FlowNet().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    psnrs = []
    times = []

    print(f"Benchmarking {len(dataset)} samples on 9070 XT...")

    with torch.no_grad():
        for i, (im1, im3, im2) in enumerate(loader):
            im1, im3, im2 = im1.to(device), im3.to(device), im2.to(device)

            # Warm-up check for first 10 frames to ignore initialization spikes
            start_time = time.time()
            pred = model(im1, im3)
            # Sync GPU to get accurate timing on AMD
            torch.cuda.synchronize() 
            elapsed = time.time() - start_time

            if i > 10: # Ignore warm-up frames
                times.append(elapsed)
                psnrs.append(calculate_psnr(pred, im2))

            if i % 100 == 0 and i > 0:
                print(f"Sample {i} | Avg PSNR: {np.mean(psnrs):.2f} dB | Avg Latency: {np.mean(times)*1000:.2f} ms")

            # Break early if you just want a quick sample, otherwise remove this
            if i == 1000: break 

    print("\n" + "="*30)
    print(f"FINAL RESULTS")
    print(f"Average PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Average Time per Frame: {np.mean(times)*1000:.2f} ms")
    print(f"Max Possible FPS: {1.0/np.mean(times):.1f} FPS")
    print("="*30)

if __name__ == "__main__":
    main()