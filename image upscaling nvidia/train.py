import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VimeoDataset
from model import ESPCN

# ==============================================================================
# ⚠️ PATHS: ADD YOUR SPECIFIC DIRECTORIES HERE ⚠️
# ==============================================================================
# Point this to the folder containing BOTH 'low_resolution' and 'target' folders
TRAIN_DATA_ROOT = r"D:\vimeo_super_resolution_test" 
SAVE_MODEL_PATH = "espcn_vimeo_final.pth"
# ==============================================================================

# Hyperparameters
BATCH_SIZE = 16  # Keep this low (8 or 16) so you don't crash your 4GB RTX 3050 VRAM
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

def train():
    # 1. Setup Device (Forces PyTorch to use your RTX 3050)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Dataset
    print("Loading dataset...")
    train_dataset = VimeoDataset(root_dir=TRAIN_DATA_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Found {len(train_dataset)} image pairs.")

    # 3. Initialize Model, Loss (L1 generally yields sharper SR results), and Optimizer
    model = ESPCN(scale_factor=4).to(device)
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            # Move images to the RTX 3050
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Forward pass
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

        # Average loss for the epoch
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {epoch_loss/len(train_loader):.6f} ---")

        # Save the model after every epoch
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train()