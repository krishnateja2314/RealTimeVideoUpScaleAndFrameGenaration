import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )
    def forward(self, x): 
        return x + self.body(x) * self.res_scale

class Upscaler(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (2**2), 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64 * (2**2), 3, padding=1),
            nn.PixelShuffle(2)
        )
        self.tail = nn.Conv2d(64, 3, 3, padding=1)
        
        # Initialize weights properly to prevent NaN explosions
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.upsample(x + res)
        return self.tail(x)