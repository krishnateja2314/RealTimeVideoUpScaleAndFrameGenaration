import torch
import math
import os
import sys
from torch.utils.data import DataLoader
from dataset import VimeoDataset
from model import Upscaler

# This makes CUDA errors show up immediately at the right line
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- CONFIGURATION ---
DATA_ROOT = r"D:\vimeo_super_resolution_test\vimeo_super_resolution_test"
LIST_FILE = os.path.join(DATA_ROOT, "sep_testlist.txt")
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- INIT ---
dataset = VimeoDataset(DATA_ROOT, LIST_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
model = Upscaler().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.L1Loss()

print(f"Starting training on {DEVICE}...")
print(f"Total batches per epoch: {len(loader)}")
print("-" * 40)

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    total_loss = 0
    valid_batches = 0

    for i, (lr, hr) in enumerate(loader):
        try:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)

            # Check for NaN in inputs BEFORE forward pass
            if torch.isnan(lr).any() or torch.isnan(hr).any():
                print(f"⚠️ NaN in inputs at Batch {i}! Skipping...")
                continue

            optimizer.zero_grad()
            output = model(lr)
            output = output.clamp(0, 1)
            loss = criterion(output, hr)

            # Check loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf loss at Batch {i}! Skipping...")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

        except RuntimeError as e:
            print(f"❌ Runtime error at Batch {i}: {e}")
            print("Skipping this batch and continuing...")
            optimizer.zero_grad()
            continue

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    print("=" * 40)
    print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
    print("=" * 40)

    torch.save(model.state_dict(), f"upscaler_epoch{epoch+1}.pth")
    print(f"💾 Model saved as upscaler_epoch{epoch+1}.pth")

torch.save(model.state_dict(), "upscaler_vimeo_final.pth")
print("🎉 Training Complete! Final model saved as upscaler_vimeo_final.pth")