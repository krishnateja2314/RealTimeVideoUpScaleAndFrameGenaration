import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VimeoDataset
from model import UltraFastUpscaler
import os

# --- CONFIGURATION (Safe to be outside) ---
DATA_ROOT = r"D:\vimeo_super_resolution_test"
LIST_FILE = os.path.join(DATA_ROOT, "sep_testlist.txt")
BATCH_SIZE = 8
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🛡️ THE WINDOWS MULTIPROCESSING SHIELD 🛡️
if __name__ == '__main__':
    
    # --- INIT ---
    dataset = VimeoDataset(DATA_ROOT, LIST_FILE)
    
    # num_workers=4 makes data loading fast, but REQUIRES the if __name__ == '__main__' block on Windows
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = UltraFastUpscaler().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss() 

    print(f"🚀 Starting training on {DEVICE}...")

    # --- LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            
            # Bulletproof NaN Shield
            if torch.isnan(loss):
                print(f"⚠️ DANGER: NaN detected at Batch {i}. Skipping...")
                continue 
                
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i:04d} | Loss: {loss.item():.4f}")

        # Save model every epoch
        torch.save(model.state_dict(), f"fast_upscaler_ep{epoch}.pth")
        print(f"💾 Saved epoch {epoch}")