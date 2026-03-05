from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    # 直接使用原始的 ENU 座標，不再進行中值中心化與縮放
    # means = means - means.median(dim=0).values
    # scale_factor = 1
    # means = means / scale_factor
    # scales = scales / scale_factor

    # 確保資料類型正確
    if means.dtype == torch.float64:
        means = means.to(torch.float32)

    # 移除所有旋轉與座標軸變換 (原本是為了外接視覺化工具而做的 swizzle)
    # 不再進行 rotation = adjustment @ rotation 等操作
    
    # 處理旋轉四元數 (Quaternion)，從 xyzw 轉為 wxyz (PLY 常見格式)
    rotations_np = rotations.detach().cpu().numpy()
    rotations_obj = R.from_quat(rotations_np)
    # 直接獲取原始旋轉，不需要乘上任何 rotation 矩陣
    rotations_matrix = rotations_obj.as_matrix()
    rotations_final = R.from_matrix(rotations_matrix).as_quat()
    
    x, y, z, w = rearrange(rotations_final, "g xyzw -> xyzw g")
    rotations_formatted = np.stack((w, x, y, z), axis=-1)

    # 由於我們的軸是為了球諧函數而切換的，我們僅導出 DC 頻帶
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(), # nx, ny, nz
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations_formatted,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
