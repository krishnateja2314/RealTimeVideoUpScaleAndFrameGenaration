import torch
from torch.utils.data import DataLoader
import time
from dataset import VimeoDataset
from model import FlowNet

DATA_ROOT = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/sequences"
TRAIN_LIST = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/tri_trainlist.txt"

def main():
    print("Starting training...")
    # On ROCm, torch.cuda.is_available() will return True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    dataset = VimeoDataset(DATA_ROOT, TRAIN_LIST)
    loader = DataLoader(
        dataset,
        batch_size=16,      # You can likely handle 16 or 32 with your VRAM
        shuffle=True,
        num_workers=4,      # Uses multiple CPU cores to load images
        pin_memory=True     # Speeds up transfer from RAM to GPU
    )

    model = FlowNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        print(f"\nEpoch {epoch}")
        for i, (im1, im3, im2) in enumerate(loader):
            im1, im3, im2 = im1.to(device), im3.to(device), im2.to(device)

            optimizer.zero_grad()
            pred = model(im1, im3)
            
            loss = torch.nn.functional.l1_loss(pred, im2)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Step {i} | Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), "model.pth")
        print("Model saved!")

if __name__ == "__main__":
    main()