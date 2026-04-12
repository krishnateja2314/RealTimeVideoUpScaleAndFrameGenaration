import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3):
        super(ESPCN, self).__init__()
        
        # Calculate the required output channels for PixelShuffle
        out_channels = num_channels * (scale_factor ** 2)
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return torch.clamp(x, 0.0, 1.0)