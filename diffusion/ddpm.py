import torch, torch.nn as nn

class SimpleDDPMUNet(nn.Module):
    """아주 단순한 UNet 대체 (실사용 시 legacy 코드로 교체 권장).
    pretrained checkpoint의 구조가 다르면 로드 실패합니다.
    """
    def __init__(self, in_channels=3, out_channels=3, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channels, 3, 1, 1),
        )
    def forward(self, x, t=None):
        return self.net(x)

class DDPMModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ch=64, **kwargs):
        super().__init__()
        self.unet = SimpleDDPMUNet(in_channels, out_channels, ch)
    def forward(self, x, t=None):
        return self.unet(x, t)
