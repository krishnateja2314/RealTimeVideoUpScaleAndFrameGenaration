import torch
import torch.nn as nn


# ----------------------------------------
# Channel Attention (RCAN idea)
# ----------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


# ----------------------------------------
# Residual Attention Block
# ----------------------------------------
class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        self.ca = ChannelAttention(channels)

    def forward(self, x):
        res = self.block(x)
        res = self.ca(res)
        return x + res * 0.2   # residual scaling (EDSR trick)


# ----------------------------------------
# Residual-in-Residual Group
# ----------------------------------------
class RRDB(nn.Module):
    def __init__(self, channels, num_blocks=3):
        super().__init__()

        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return x + self.blocks(x) * 0.2


# ----------------------------------------
# REAL-TIME ADVANCED UPSCALER
# ----------------------------------------
class RealTimeUpscaler(nn.Module):
    def __init__(self, scale_factor=2, num_rrdb=6):
        super().__init__()

        channels = 64

        # shallow features
        self.head = nn.Conv2d(3, channels, 3, padding=1)

        # deep feature extraction
        self.body = nn.Sequential(
            *[RRDB(channels) for _ in range(num_rrdb)]
        )

        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1)

        # Upsampling (PixelShuffle)
        up_layers = []
        for _ in range(scale_factor // 2):
            up_layers += [
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.upsample = nn.Sequential(*up_layers)

        # reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1)
        )

    def forward(self, x):

        shallow = self.head(x)

        deep = self.body(shallow)
        deep = self.body_conv(deep)

        features = shallow + deep

        up = self.upsample(features)

        out = self.tail(up)

        return out