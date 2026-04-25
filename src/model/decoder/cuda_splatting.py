from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points
from ..encoder.costvolume.conversions import depth_to_relative_disparity
import torch.nn.functional as F
import torchvision
def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result
def get_projection_matrix_asymmetric(K, near, far, W, H):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    (b,) = near.shape
    res = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    res[:, 0, 0] = 2 * fx / W
    res[:, 1, 1] = 2 * fy / H
    res[:, 0, 2] = (2 * cx / W) - 1.0 # 關鍵：非 0
    res[:, 1, 2] = (2 * cy / H) - 1.0 # 關鍵：非 0
    res[:, 2, 2] = far / (far - near)
    res[:, 2, 3] = -(far * near) / (far - near)
    res[:, 3, 2] = 1.0
    return res

def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
    skew_params: Float[Tensor, "batch 4"] | None = None,
    crop_offsets: Float[Tensor, "batch 2"] | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Cast all inputs to float32 to ensure compatibility with C++ rasterizer, 
    # which does not support half precision (AMP).
    extrinsics = extrinsics.float()
    intrinsics = intrinsics.float()
    near = near.float()
    far = far.float()
    background_color = background_color.float()
    gaussian_means = gaussian_means.float()
    
    gaussian_covariances = gaussian_covariances.float()
    gaussian_sh_coefficients = gaussian_sh_coefficients.float()
    gaussian_opacities = gaussian_opacities.float()
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    # When crop_offsets is provided: render full 2048×2048, then unwarp + crop.
    # Otherwise: render directly at the requested output size.
    h_out, w_out = image_shape
    
    if crop_offsets is not None:
        h_render, w_render = 2048, 2048
    else:
        h_render, w_render = h_out, w_out
    fov_x, fov_y = get_fov(intrinsics, h_render).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    projection_matrix = get_projection_matrix_asymmetric(intrinsics, near, far, w_render, h_render)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    all_images = []
    for i in range(b):
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h_render,
            image_width=w_render,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        image, _ = rasterizer(
            means3D=gaussian_means[i].float(),
            means2D=mean_gradients.float(),
            shs=shs[i].float() if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :].float(),
            opacities=gaussian_opacities[i, ..., None].float(),
            cov3D_precomp=gaussian_covariances[i, :, row, col].float(),
        )
        # Apply inverse skew warp: rendered (s=0) → original satellite geometry (s≠0)
        if skew_params is not None:
            s_i    = skew_params[i, 0].item()
            fy_i   = skew_params[i, 1].item()
            cy_i   = skew_params[i, 2].item()
            dcx_i  = skew_params[i, 3].item()
            v_c = torch.arange(h_render, dtype=torch.float32, device=image.device)
            u_c = torch.arange(w_render, dtype=torch.float32, device=image.device)
            vv, uu = torch.meshgrid(v_c, u_c, indexing='ij')
            u_sample = uu + dcx_i - (s_i / fy_i) * (vv - cy_i)
            v_sample = vv
            u_norm = (u_sample / (w_render - 1)) * 2.0 - 1.0
            v_norm = (v_sample / (h_render - 1)) * 2.0 - 1.0
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
            image = F.grid_sample(
                image.unsqueeze(0), grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            ).squeeze(0)

        # Crop to output size
        if crop_offsets is not None:
            col_s = int(crop_offsets[i, 0].item())
            row_s = int(crop_offsets[i, 1].item())
            # Keep output size fixed even when offsets fall outside render bounds.
            # This mirrors dataset-side padding behavior used during patch generation.
            x1, y1 = col_s, row_s
            x2, y2 = col_s + w_out, row_s + h_out

            read_x1, read_y1 = max(0, x1), max(0, y1)
            read_x2, read_y2 = min(w_render, x2), min(h_render, y2)

            read_w = read_x2 - read_x1
            read_h = read_y2 - read_y1

            cropped = image.new_zeros((image.shape[0], h_out, w_out))
            if read_w > 0 and read_h > 0:
                out_x1 = read_x1 - x1
                out_y1 = read_y1 - y1
                cropped[:, out_y1 : out_y1 + read_h, out_x1 : out_x1 + read_w] = \
                    image[:, read_y1:read_y2, read_x1:read_x2]
            image = cropped
        all_images.append(image)
    return torch.stack(all_images)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)
