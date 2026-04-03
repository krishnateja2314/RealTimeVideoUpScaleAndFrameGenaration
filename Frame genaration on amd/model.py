import torch
import torch.nn as nn
import torch.nn.functional as F

def warp(x, flow):
    # Generates a grid to sample pixels from
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)
    
    vgrid = grid + flow
    
    # Scale grid to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    return output

class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple Encoder-Decoder to predict flow
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 5, 4, 2, 1) # 4 channels for flow (dx,dy for im1 and im3), 1 for mask
        )

    def forward(self, im1, im3):
        x = torch.cat((im1, im3), dim=1)
        x = self.encoder(x)
        out = self.decoder(x)
        
        flow1 = out[:, 0:2, :, :]
        flow3 = out[:, 2:4, :, :]
        mask = torch.sigmoid(out[:, 4:5, :, :])
        
        warped_im1 = warp(im1, flow1)
        warped_im3 = warp(im3, flow3)
        
        # Blend the two warped images using the mask
        pred_im2 = mask * warped_im1 + (1 - mask) * warped_im3
        return pred_im2