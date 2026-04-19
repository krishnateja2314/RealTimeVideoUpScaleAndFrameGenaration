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

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class InterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Wider Encoder for superior motion feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(0.1, True),
            ResBlock(256),
            ResBlock(256)
        )
        
        # 2. Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 5, 3, 1, 1) # 4 channels for flow (dx,dy for im1 and im3), 1 for mask
        )
        
        # 3. The Quality Refiner (Fixes ghosting and sharpens edges)
        self.refine = nn.Sequential(
            nn.Conv2d(9, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), # Inputs: im1, pred_im2, im3
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, im1, im3):
        # Extract features and predict flow/mask
        x = torch.cat((im1, im3), dim=1)
        feat = self.encoder(x)
        out = self.decoder(feat)
        
        flow1 = out[:, 0:2, :, :]
        flow3 = out[:, 2:4, :, :]
        mask = torch.sigmoid(out[:, 4:5, :, :])
        
        # Warp the images
        warped_im1 = warp(im1, flow1)
        warped_im3 = warp(im3, flow3)
        
        # Base Prediction
        base_pred = mask * warped_im1 + (1 - mask) * warped_im3
        
        # Synthesis/Refinement Pass
        # Network looks at the base prediction in context of the original frames
        refine_input = torch.cat((im1, base_pred, im3), dim=1)
        residual = self.refine(refine_input)
        
        # Add residual details and clamp to valid image range
        final_pred = torch.clamp(base_pred + residual, 0, 1)
        return final_pred