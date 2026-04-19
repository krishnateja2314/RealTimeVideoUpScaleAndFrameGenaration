import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import VimeoDataset
from model import InterpolationModel

# Charbonnier Loss (Better than L1 for edges)
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

# Laplacian Pyramid Loss (Focuses on sharpness)
class LapLoss(nn.Module):
    def __init__(self):
        super(LapLoss, self).__init__()
        self.kernel = torch.tensor([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], 
                                    [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]).float() / 256.0
        self.kernel = self.kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)

    def pyramid(self, x):
        padding = 2
        low = F.conv2d(x, self.kernel.to(x.device), padding=padding, groups=3)
        return x - low

    def forward(self, x, y):
        return F.l1_loss(self.pyramid(x), self.pyramid(y))

def main():
    device = torch.device('cuda')
    DATA_ROOT = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/sequences"
    TRAIN_LIST = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/tri_trainlist.txt"

    dataset = VimeoDataset(DATA_ROOT, TRAIN_LIST)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True)

    
    model = InterpolationModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    criterion_pixel = CharbonnierLoss().to(device)
    criterion_lap = LapLoss().to(device)

    for epoch in range(30):
        for i, (im1, im3, im2) in enumerate(loader):
            im1, im3, im2 = im1.to(device), im3.to(device), im2.to(device)

            pred = model(im1, im3)
            
            # Combine losses: Pixel-wise + Sharpness
            loss = criterion_pixel(pred, im2) + (0.5 * criterion_lap(pred, im2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()