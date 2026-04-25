import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the DAMV2 package is importable regardless of working directory.
_DAMV2_DIR = Path(__file__).parent / "Depth-Anything-V2"
if str(_DAMV2_DIR) not in sys.path:
    sys.path.insert(0, str(_DAMV2_DIR))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

# ImageNet normalization constants used by DAMV2
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class FrozenDAMV2(nn.Module):
    """Frozen Depth Anything V2 wrapper for batch tensor inference.

    Handles ImageNet normalization and resolution alignment (must be multiples
    of 14 for the ViT patch embedding) internally.  The model weights are never
    updated during training.

    Args:
        encoder:    ViT size key — "vits" | "vitb" | "vitl" | "vitg"
        checkpoint: Path to the .pth weight file.
    """

    def __init__(self, encoder: str, checkpoint: str) -> None:
        super().__init__()
        self.model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    # Override train() so Lightning's .train() call never puts DAMV2 back into
    # training mode (which would activate BatchNorm running-stat updates).
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen DAMV2 inference on a batch of RGB images.

        Args:
            x: [B, 3, H, W]  float32, values in [0, 1], RGB channel order.

        Returns:
            depth: [B, H, W]  relative depth (larger value = farther from
                   camera), same spatial resolution as input.
        """
        B, C, H, W = x.shape

        # 1. Resize to the largest multiple of 14 that fits in the input size.
        #    256 → 252  (18 × 14),  but keep original H/W for the output.
        H14 = (H // 14) * 14
        W14 = (W // 14) * 14
        x_resized = F.interpolate(
            x, size=(H14, W14), mode="bilinear", align_corners=False
        )

        # 2. ImageNet normalization (DAMV2 was pre-trained with these stats).
        x_norm = (x_resized - self.mean.to(x.device)) / self.std.to(x.device)

        # 3. DAMV2 forward → [B, H14, W14]
        depth = self.model(x_norm)

        # 4. Restore original spatial size.
        if H14 != H or W14 != W:
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)

        return depth  # [B, H, W]
