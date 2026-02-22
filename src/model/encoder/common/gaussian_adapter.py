from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        # RPC 模式：scale 直接代表公尺
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        
        # SH 處理：
        # 3DGS 的顏色公式: color = 0.5 + C0 * sh_dc, 其中 C0 ≈ 0.28
        # 若要讓 color 覆蓋 [0, 1]，sh_dc 需要在 [-1.77, 1.77] 範圍
        # 
        # 方案: 直接縮放 DC component (不使用 tanh，讓 gradient 更順暢)
        # 網路初始輸出接近 0，乘以 3.0 讓梯度更大，學習更快
        sh_dc = sh[..., :1]  # DC component [B, N, 3, 1]
        sh_higher = sh[..., 1:]  # Higher order [B, N, 3, d_sh-1]
        
        # DC: 使用 tanh 縮放到 [-0.5, 0.5]，再加 0.5 移到 [0, 1]
        # 這比 sigmoid 有更好的梯度特性：
        #   - sigmoid 在輸入遠離 0 時梯度很小
        #   - tanh 在 [-2, 2] 範圍都有不錯的梯度
        # 同時添加 bias=0.5 確保初始渲染是灰色
        sh_dc_scaled = torch.tanh(sh_dc) * 0.5 + 0.5  # 輸出 [0, 1]
        
        # Higher order: 保持較小的值（視角相關效果通常較弱）
        sh = torch.cat([sh_dc_scaled, sh_higher], dim=-1)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        

        # Force all Gaussians to have scale = 0.1 meters
        # scales = torch.full_like(scales, 0.07)  # All axes = 0.1 meter

        # 讓球全部都白色:
        # For SH degree 0 (DC component), white = 0.28209 (1/(2*sqrt(pi)))
        # All higher-degree coefficients = 0
        # sh_shape = (*opacities.shape, 3, self.d_sh)
        # sh = torch.zeros(sh_shape, dtype=sh.dtype, device=sh.device)
        # # Set DC component (index 0) for RGB channels to white
        # sh[..., :, 0] = 0.28209  # This makes the color white when rendered




        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
