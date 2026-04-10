import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, scale_factor=4):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x