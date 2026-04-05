import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import RealTimeUpscaler
from dataset import VimeoDataset


# Prevent dynamo crashes (Windows safe)
torch._dynamo.config.suppress_errors = True


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # Dataset
    # =====================
    DATASET_DIR = r"D:\vimeo_super_resolution_test\vimeo_super_resolution_test"

    dataset = VimeoDataset(DATASET_DIR)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # =====================
    # Model
    # =====================
    model = RealTimeUpscaler(upscale_factor=4).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda")


    epochs = 50

    # =====================
    # Training Loop
    # =====================
    for epoch in range(epochs):

        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for lr_imgs, hr_imgs in loop:

            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                sr_imgs = model(lr_imgs)

                # ✅ clamp ONLY during training output
                sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)

                loss = criterion(sr_imgs, hr_imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    print("Training complete!")


if __name__ == "__main__":
    train()

        