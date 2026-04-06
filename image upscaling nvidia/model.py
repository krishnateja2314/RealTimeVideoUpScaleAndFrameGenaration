import torch
import torch.nn as nn

class FastResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_scale = 0.1 
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x): 
        return x + self.body(x) * self.res_scale

class UltraFastUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Feature Extraction: NOW ACCEPTS 21 CHANNELS (7 frames * RGB)
        self.head = nn.Conv2d(21, 32, kernel_size=3, padding=1)
        
        # 2. AI Processing
        self.body = nn.Sequential(*[FastResidualBlock(32) for _ in range(3)])
        
        # 3. Fixed AI Upscale (PixelShuffle directly to 4x)
        self.upsample = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.PixelShuffle(4) 
        )

    def forward(self, x):
        features = self.head(x)
        features = features + self.body(features)
        out_4x = self.upsample(features)
        return out_4x