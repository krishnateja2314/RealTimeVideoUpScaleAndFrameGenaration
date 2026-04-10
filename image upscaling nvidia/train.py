import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import DIV2KDataset
from model import ESPCN
model = ESPCN(scale_factor=4)

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
HR_DIR = r"D:\DIV2K\DIV2K_train_HR"
LR_DIR = r"D:\DIV2K\X4"


# -------------------------
# Train One Batch
# -------------------------
def train_step(lr, hr, model, optimizer, criterion):

    lr = lr.to(device)
    hr = hr.to(device)

    sr = model(lr)
    loss = criterion(sr, hr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# -------------------------
# Train One Epoch
# -------------------------
def train_epoch(loader, model, optimizer, criterion, epoch):

    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch+1}")

    total_loss = 0

    for lr, hr in loop:
        loss = train_step(lr, hr, model, optimizer, criterion)

        total_loss += loss
        loop.set_postfix(loss=loss)

    return total_loss / len(loader)


# -------------------------
# Main Function
# -------------------------
def main():

    # Dataset
    dataset = DIV2KDataset(HR_DIR, LR_DIR)

# 🔍 DEBUG CHECK (add this)
    lr, hr = dataset[0]
    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)

# Then continue normally
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

   

    # Model
    model = ESPCN(scale_factor=4).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    best_loss = float("inf")

    for epoch in range(epochs):

        avg_loss = train_epoch(loader, model, optimizer, criterion, epoch)

        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "espcn.pth")
            print("✅ Model Saved")


    print("Training finished!")


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    main()