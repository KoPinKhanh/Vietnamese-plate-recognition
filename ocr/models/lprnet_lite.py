"""
LPRNet-Lite Architecture for Vietnamese License Plate OCR.

Lightweight, fully-convolutional, CTC-compatible network tailored for
CPU inference on edge devices (Raspberry Pi 5) via OpenCV DNN.

Architecture design:
- Input shape: [B, 3, 48, 160] (NCHW)
- Feature extraction: Hierarchical 2D convolutions with BatchNorm and ReLU
- Multi-scale fusion: Concatenates multi-resolution spatial features (early, mid, late)
- Temporal downsampling: Width is downsampled by factor 4 (160 -> 40 time steps)
- Vertical collapse: Height is progressively collapsed (48 -> 24 -> 12 -> 6 -> 3 -> 1)
- Output shape: [B, num_classes, 40] where num_classes = 37 (36 alphanumeric + 1 CTC blank)
- Total parameters: ~0.80M (very fast, low memory)
"""

import torch
import torch.nn as nn
from ocr.charset import NUM_CLASSES


class LPRNetLite(nn.Module):
    """
    LPRNet-Lite model for Vietnamese license plate sequence recognition with CTC.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: Input (3, 48, 160) -> (64, 24, 80)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 24, 80)
        )

        # Stage 2: (64, 24, 80) -> (128, 12, 40)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 12, 40)
        )

        # Stage 3: (128, 12, 40) -> (256, 1, 40)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (128, 6, 40)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (256, 3, 40)
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(1, 1), padding=0),  # (256, 1, 40)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Multi-scale feature adapters to fuse early, mid, and deep features at spatial size (1, 40)
        # Stage 1 adapter: (64, 24, 80) -> AvgPool -> (64, 1, 40) -> Conv 1x1 -> (64, 1, 40)
        self.adapter1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(24, 2), stride=(24, 2)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 2 adapter: (128, 12, 40) -> AvgPool -> (128, 1, 40) -> Conv 1x1 -> (64, 1, 40)
        self.adapter2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(12, 1), stride=(12, 1)),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final classifier: fuses adapter1(64) + adapter2(64) + stage3(256) = 384 channels
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(384, num_classes, kernel_size=1)  # (num_classes, 1, 40)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor of shape [B, 3, 48, 160].
            
        Returns:
            Logits tensor of shape [B, num_classes, 40] ([B, C, T]).
        """
        f1 = self.stage1(x)       # [B, 64, 24, 80]
        f2 = self.stage2(f1)      # [B, 128, 12, 40]
        f3 = self.stage3(f2)      # [B, 256, 1, 40]

        a1 = self.adapter1(f1)    # [B, 64, 1, 40]
        a2 = self.adapter2(f2)    # [B, 64, 1, 40]

        # Multi-scale concatenation
        fused = torch.cat([a1, a2, f3], dim=1)  # [B, 384, 1, 40]
        logits_4d = self.classifier(fused)       # [B, num_classes, 1, 40]
        
        # Squeeze height dimension (1) to produce [B, C, T]
        logits = logits_4d.squeeze(2)            # [B, num_classes, 40]
        return logits


def build_lprnet_lite(num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3) -> LPRNetLite:
    """Helper factory function to create LPRNetLite instance."""
    return LPRNetLite(num_classes=num_classes, dropout_rate=dropout_rate)
