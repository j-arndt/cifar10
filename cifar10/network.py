"""ResNet-9 for CIFAR-10 speedrun."""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )

    def forward(self, x):
        return x + self.net(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.prep   = ConvBNReLU(3, 64)
        self.layer1 = nn.Sequential(ConvBNReLU(64, 128), nn.MaxPool2d(2), ResidualBlock(128))
        self.layer2 = nn.Sequential(ConvBNReLU(128, 256), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(ConvBNReLU(256, 512), nn.MaxPool2d(2), ResidualBlock(512))
        self.pool   = nn.MaxPool2d(4)
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet9().cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")  # ~6.6M
    out = model(torch.randn(4, 3, 32, 32).cuda())
    print(f"Output shape: {out.shape}")  # (4, 10)
