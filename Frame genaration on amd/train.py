import torch
import torch_directml
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
import time

from dataset import VimeoDataset
from model import InterpolationModel

# ----------------------------
# Paths
# ----------------------------
DATA_ROOT = "data/vimeo_triplet/sequences"
TRAIN_LIST = "data/vimeo_triplet/tri_trainlist.txt"


def main():
    print("Starting training...")

    # ----------------------------
    # Device (AMD GPU via DirectML)
    # ----------------------------
    device = torch_directml.device()
    print("Using device:", device)

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset = VimeoDataset(DATA_ROOT, TRAIN_LIST)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,   # 🔥 safest + avoids all Windows issues
        pin_memory=False
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = InterpolationModel().to(device)

    # ----------------------------
    # Optimizer (SGD → no CPU fallback)
    # ----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(10):
        print(f"\nEpoch {epoch}")

        for i, (im1, im3, im2) in enumerate(loader):
            im1 = im1.to(device)
            im3 = im3.to(device)
            im2 = im2.to(device)

            # ----------------------------
            # Forward + timing
            # ----------------------------
            start_time = time.time()

            pred = model(im1, im3)

            inference_time = time.time() - start_time

            # ----------------------------
            # Loss
            # ----------------------------
            loss = torch.mean(torch.abs(pred - im2))

            # ----------------------------
            # Backprop
            # ----------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------------------
            # Logs
            # ----------------------------
            if i % 100 == 0:
                print(
                    f"Step {i} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Time: {inference_time:.4f}s | "
                    f"Device: {im1.device}"
                )

        # ----------------------------
        # Save model
        # ----------------------------
        torch.save(model.state_dict(), "model.pth")
        print("Model saved!")

    print("Training finished!")


# ----------------------------
# Windows fix (IMPORTANT)
# ----------------------------
if __name__ == "__main__":
    freeze_support()
    main()