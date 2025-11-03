import torch, torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base*2, 3, 2, 1), nn.BatchNorm2d(base*2), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.BatchNorm2d(base*4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base*4, 1)
    def forward(self, x):
        f = self.net(x).flatten(1)
        return self.head(f).squeeze(1)
