import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4):
        super(FSRCNN, self).__init__()

        d = 56
        s = 12
        m = 4

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, d, 5, padding=2),
            nn.PReLU(d)
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.PReLU(s)
        )

        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, 3, padding=1))
            mapping_layers.append(nn.PReLU(s))

        self.mapping = nn.Sequential(*mapping_layers)

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, 1),
            nn.PReLU(d)
        )

        self.deconv = nn.ConvTranspose2d(
            d, 3, 9,
            stride=scale_factor,
            padding=3,
            output_padding=1
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x