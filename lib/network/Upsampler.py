import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
        

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        y = self.act_fn(self.conv(x))
        return y


class Upsampler(nn.Module):
    def __init__(self, input_dim, output_dim, network_capacity=32):
        super(Upsampler, self).__init__()

        dims = [network_capacity * s for s in [4, 8, 4, 2, 1]]
        
        self.inc = DoubleConv(input_dim, dims[0])
        self.down1 = Down(dims[0], dims[1])
        self.up1 = Up(dims[0] + dims[1], dims[2])
        self.up2 = Up(dims[2], dims[3])
        self.up3 = Up(dims[3], dims[4])
        self.outc = OutConv(dims[4], output_dim)

    def forward(self, x):
        x1 = self.inc(x) # res == 128
        x2 = self.down1(x1) # res == 64
        y = self.up1(x2, x1) # res == 128
        y = self.up2(y) # res == 256
        y = self.up3(y) # res == 512
        y = self.outc(y) # res == 512
        return y