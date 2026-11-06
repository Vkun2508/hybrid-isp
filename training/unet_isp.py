"""LightweightISPUNet: the team's learned RAW denoiser (copied verbatim from
training/SIDD_CC_RAW_Complete_Pipeline.ipynb, Section 10).

Input/output: [B, 4, H/2, W/2] packed Bayer [R, G1, G2, B], normalised (raw - black) / (white - black),
unclipped. Output = input + predicted residual (denoised RAW, same shape).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Lightweight 2-D convolution: 3x3 Depthwise + 1x1 Pointwise."""
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResBlock(nn.Module):
    """Residual Block with optional Depthwise-Separable Convolutions and LeakyReLU."""
    def __init__(self, channels, use_dw_sep=False):
        super().__init__()
        Conv = DepthwiseSeparableConv2d if use_dw_sep else lambda in_c, out_c: nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True)
        self.conv1 = Conv(channels, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = Conv(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class UpsampleBlock(nn.Module):
    """Sub-pixel convolution (PixelShuffle) upsampling block to prevent checkerboard artifacts."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=True)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))

class LightweightISPUNet(nn.Module):
    """
    Lightweight Residual U-Net for 4-channel Bayer RAW Smartphone Denoising.
    Suitable for mobile deployment on CPU, Apple MPS, and CUDA GPUs.
    """
    def __init__(self, in_channels=4, out_channels=4, base_channels=32, num_bottleneck_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        c = base_channels

        # Initial Feature Extraction
        self.head = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=True)

        # Encoder Path
        self.enc1 = ResBlock(c, use_dw_sep=False)
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=True)

        self.enc2 = ResBlock(c * 2, use_dw_sep=True)
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.enc3 = ResBlock(c * 4, use_dw_sep=True)
        self.down3 = nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1, bias=True)

        # Bottleneck with Depthwise-Separable Convolutions
        bottleneck_layers = [ResBlock(c * 8, use_dw_sep=True) for _ in range(num_bottleneck_blocks)]
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Decoder Path with PixelShuffle Upsampling
        self.up3 = UpsampleBlock(c * 8, c * 4)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c * 8, c * 4, kernel_size=1, bias=True),
            ResBlock(c * 4, use_dw_sep=True)
        )

        self.up2 = UpsampleBlock(c * 4, c * 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, kernel_size=1, bias=True),
            ResBlock(c * 2, use_dw_sep=True)
        )

        self.up1 = UpsampleBlock(c * 2, c)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=True),
            ResBlock(c, use_dw_sep=False)
        )

        # Output Reconstruction Head (Unconstrained Residual Prediction)
        self.tail = nn.Conv2d(c, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # 1. Variable Image Size Support: Pad to multiple of 8
        H, W = x.shape[2], x.shape[3]
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_pad = x

        # 2. Encoder
        f0 = self.head(x_pad)
        e1 = self.enc1(f0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        # 3. Bottleneck
        b = self.bottleneck(d3)

        # 4. Decoder with Skip Connections
        u3 = self.up3(b)
        cat3 = torch.cat([u3, e3], dim=1)
        dec3 = self.dec3(cat3)

        u2 = self.up2(dec3)
        cat2 = torch.cat([u2, e2], dim=1)
        dec2 = self.dec2(cat2)

        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, e1], dim=1)
        dec1 = self.dec1(cat1)

        # 5. Residual Prediction
        residual = self.tail(dec1)

        # 6. Remove Padding
        if pad_h > 0 or pad_w > 0:
            residual = residual[:, :, :H, :W]

        # 7. Global Residual Learning: Clean = Noisy + Predicted Residual
        clean = x + residual
        return clean
