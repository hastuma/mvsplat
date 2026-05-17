# SkySplat: Feed-Forward 3D Gaussian Splatting from Satellite Imagery

**SkySplat** is a research project that adapts [MVSplat](https://github.com/donydchen/mvsplat) (ECCV 2024 Oral) for satellite remote sensing. Instead of ground-level perspective images, SkySplat takes **satellite image patches with RPC (Rational Polynomial Coefficients) metadata** as input and reconstructs geo-aligned 3D Gaussian Splatting (3DGS) representations in a single forward pass — no per-scene optimization required.

> **Status**: Active training. Core rendering pipeline is working. The dark-hole rendering artifact (previously the primary blocker) has been resolved via inverse skew warp correction.

---

## Key Differences from MVSplat

| Aspect | MVSplat | SkySplat |
|---|---|---|
| Input | Ground-level RGB images | Satellite image patches |
| Camera model | Perspective pinhole | RPC (Rational Polynomial Coefficients) |
| Camera parameters | Learned / dataset-provided | Read from pre-computed JSON camera files |
| Coordinate system | Camera-centric | WGS84 → ENU local tangent plane |
| Depth axis | Distance from camera | Altitude (MSL) converted to camera distance |
| Cost volume | Epipolar plane sweep | Altitude-range plane sweep via RPC warping |
| Skew correction | N/A | Post-render inverse warp for skewed satellite sensors |

---

## Pipeline

### 1. Input & Preprocessing

- **Input**: $N$ satellite image patches, each cropped to a fixed resolution (e.g., 256×256). Training uses **3 context views only** (no target); validation/test additionally hold out 1 target view for novel-view metrics.
- **Metadata**: Each patch is accompanied by its **RPC file** — 80 coefficients encoding the non-linear mapping from (lon, lat, height) → (col, row) in the original full-resolution image.
- **Patch Offset**: The pixel offset of the patch within the original full image is recorded to enable principal-point correction.
- **ENU Origin**: A shared geographic reference point (lat₀, lon₀, h₀) is assigned per scene, used as the origin of the local ENU coordinate system.

### 2. RPC Geometry & Virtual Camera Construction

The RPC model is highly non-linear and cannot be used directly by the standard 3DGS rasterizer. SkySplat approximates the RPC as a **linear pinhole camera** over the local patch region:

1. **Jacobian Linearization**: The RPC projection function is linearized around the scene center altitude to derive an approximate intrinsic matrix $K$ and extrinsic $[R|t]$.
2. **Principal Point Correction**: The projection residual between the true RPC-projected scene center and the patch image center is used to correct $(c_x, c_y)$ in $K$, ensuring the scene center reprojects exactly to the image center (128, 128).
3. **Asymmetric Frustum**: Because the principal point may lie far outside the patch image bounds, an **asymmetric projection matrix** is constructed to handle off-center rendering correctly.
4. **Virtual Camera Distance**: The approximate distance from the camera to the ground plane is derived from the RPC GSD (Ground Sampling Distance) and focal length, used to define per-view near/far bounds.

The resulting $(K, [R|t])$ pairs are stored in pre-computed JSON camera files (one per image) and loaded at training time from `{dataset_root}/camera_pose/{scene_prefix}_cameras/{image_name}.json`.

### 3. Skew Correction (Satellite Sensor Geometry)

Satellite push-broom sensors produce images with a non-zero skew parameter $s$ in the intrinsic matrix $K$, meaning the pixel grid is not perfectly rectangular. The standard 3DGS rasterizer assumes $s = 0$.

SkySplat handles this with a two-camera approach:

1. **Skew-Free Rendering Camera** (`K_s0`): Rendering is performed using a skew-corrected intrinsic matrix where $s = 0$ is enforced. This produces a geometrically correct but "straightened" render.
2. **Post-Render Inverse Warp**: After rasterization, a pixel-level inverse warp maps the rendered image back to the original skewed coordinate space so it matches the ground-truth satellite image. The warp formula per pixel $(u, v)$ is:
   $$u_\text{sample} = u + \Delta c_x - \frac{s}{f_y}(v - c_y)$$
   where $s, f_y, c_y$ come from the original (skewed) camera $K_\text{orig}$, and $\Delta c_x = c_{x,s0} - c_{x,\text{orig}}$ is the principal-point shift introduced by skew removal.

The original skewed camera parameters are stored in `{dataset_root}/camera_pose/skewed_camera/{image_name}.json` and are only used to compute the four warp parameters $(s, f_y, c_y, \Delta c_x)$ passed to the CUDA renderer.

**This fix resolved the dark-hole rendering artifacts** that were the primary blocker: the holes were caused by the rendered geometry being in skew-free coordinates while ground-truth images were in the original skewed frame.

### 4. Cost Volume Construction & Height Estimation

- **Altitude-Range Plane Sweep**: Instead of sweeping over disparity or depth, SkySplat sweeps over a predefined **altitude range** (e.g., HEIGHT_OFF ± 20 m MSL).
- **RPC Warping**: For each altitude hypothesis $H_k$, source patches are warped into the reference view using the RPC inverse projection, constructing a **4D Cost Volume** of shape `[B, V, D, H, W]`.
- **Height Probability**: A UNet-based CNN processes the cost volume and predicts a per-pixel probability distribution over altitude candidates.
- **Best Height $H^*$**: The argmax (or soft-argmax) of the distribution gives the estimated altitude per pixel.

### 5. 3D Gaussian Parameter Prediction

- **Anchor Position**: Using $H^*$ from Step 4, the RPC inverse transform maps each pixel $(u, v)$ + altitude $H^*$ to a geographic coordinate (lon, lat, H) in WGS84. This serves as the **fixed position anchor** for each Gaussian.
- **Residual Learning (Encoder)**: The MVSplat encoder backbone (UniMatch-pretrained multi-view Transformer) predicts:
  - **$\Delta \text{Pos}$** — sub-pixel position residuals to refine alignment.
  - **Scale, Opacity** — geometric footprint and visibility.
  - **SH Coefficients** — view-dependent appearance encoded as Spherical Harmonics.
- **XY Freeze**: In RPC mode, the $(x, y)$ offset within each pixel is frozen to 0.5 (pixel center), so only the RPC inverse defines the lateral position. Only the depth/altitude axis is learned.

### 6. Coordinate Realignment: WGS84 → ENU

After obtaining (lat, lon, H) for each Gaussian, positions are transformed into the **ENU (East-North-Up)** local tangent plane coordinate system to avoid floating-point precision loss at large geographic coordinates:

1. **ECEF Conversion**: Both the Gaussian positions and the ENU origin are converted from geodetic (lat, lon, h) to ECEF $(X, Y, Z)$ using the WGS84 ellipsoid model.
2. **ENU Rotation**: The ECEF difference vector is rotated using the reference point's rotation matrix into $(E, N, U)$ coordinates.
3. **Scene Centering**: All Gaussian positions are expressed relative to the scene center, with $Z$ pointing up (perpendicular to the local horizontal plane).

The target camera parameters (extrinsics) are computed in the same ENU frame, ensuring Gaussians and cameras share a consistent coordinate system.

### 7. Differentiable Rendering & Optimization

- **3DGS Rasterizer**: The geo-aligned Gaussians are rendered using the standard differentiable 3DGS rasterizer with the ENU-frame camera parameters. During training, Gaussians are rendered back at the **context** cameras (V=3) — there is no held-out target view at train time. Near/far for this render uses a wide **`dist ± 50000m`** sweep so all Gaussians stay inside the rasterizer frustum given satellite-scale camera distances.
- **Loss Functions**:
  - **$L_\text{RGB}$**: Photometric reconstruction loss (MSE + LPIPS) between rendered and **context** images, averaged over V=3 context views (self-supervised reconstruction of the input views). Validation/test still measure novel-view quality on a held-out target view via `val/lpips_val`, `val/ssim_val`, `val/mse_val`.
  - **$L_\text{Smooth}$**: Regularization on $\Delta \text{Pos}$ to prevent geometry collapse.
  - **$L_\text{Opacity}$**: Penalty when mean opacity falls below 0.3, preventing Gaussians from collapsing to transparent.
  - **$L_\text{DAMV2}$**: Pearson correlation loss between cost-volume depth and frozen Depth Anything V2 (DAMv2) relative depth. DAMv2 acts as a monocular depth teacher (weights frozen); only the encoder depth receives gradients. Controlled by `damv2_loss_weight` (default 0.1) and `damv2_loss_warmup_steps` (default 500) in the experiment config.

    **Computation details:**

    For each context view $v$, the loss is:
    $$L_\text{DAMV2} = -\frac{1}{V} \sum_{v=1}^{V} \rho\!\left(\text{cv\_depth}_v,\ -\text{DAMv2}(\text{img}_v)\right)$$

    where $\rho$ is the Pearson correlation coefficient:
    $$\rho(X, Y) = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sqrt{\sum(x_i-\bar x)^2} \cdot \sqrt{\sum(y_i-\bar y)^2} + \varepsilon}$$

    **Sign convention (critical):** DAMv2 was trained on ground-perspective images. When applied to top-down satellite images, it assigns *larger* relative-depth values to buildings (interpreting complex building-roof textures as "distant" background). The cost-volume depth, however, assigns *smaller* metric distances to buildings (buildings are physically closer to the satellite). Therefore DAMv2's output is **negated** before computing Pearson, so both signals point in the same direction (small = building = close to satellite).

    **Interpreting `loss/damv2_pearson` on WandB:**

    | Value | $\rho$ | Meaning |
    |---|---|---|
    | **−0.7 to −0.9** | +0.7–+0.9 | ✅ **Ideal** — strong positive correlation; depth structure matches DAMv2 |
    | **≈ 0** | ≈ 0 | ❌ **Failure** — depth std → 0 (flat plane collapse); Pearson undefined → returns 0 |
    | **> 0** | negative | ❌ Sign convention wrong; negate DAMv2 output |

    The loss being very negative (e.g. −0.9) is the **goal**, not a problem. A sudden jump from −0.9 to ≈ 0 indicates depth has collapsed to a flat plane (zero variance → Pearson numerically → 0), NOT that training stabilised.

---

## Camera File Structure

```
{dataset_root}/
├── camera_pose/
│   ├── {scene_prefix}_cameras/
│   │   └── {image_name}.json     ← skew-corrected (s=0) K + W2C  [used for rendering]
│   └── skewed_camera/
│       └── {image_name}.json     ← original K (s≠0)              [used for warp params only]
└── enu_origin/
    └── {scene_prefix}.json       ← [lat0, lon0, h0] ENU reference point
```

---

## Training Monitoring

During training the following artifacts are saved under `outputs/{date}/{run}/`:

| Path | Content | Frequency |
|---|---|---|
| `vis_loss/step_*.png` | Rendered vs GT side-by-side | Every 10 steps |
| `vis_depth/step_*.png` | Low-res + high-res depth maps | Every 10 steps |
| `vis_feature/step_*.png` | CNN feature map (first 3 channels) | Every 10 steps |
| `vis_context/step_*_ctx*.png` | Context view rendered vs GT | Every 100 steps |
| `gaussians_ply/step_*.ply` | 3D Gaussian cloud (open in Gaussian Viewer) | Every 50 steps |
| `gaussians_ply/step_*_full.pt` | Full Gaussian data (PyTorch) | Every 50 steps |
| `training_debug.log` | RPC coefficients, Gaussian stats, near/far | Every 10 steps |

WandB logs:

| Key | Phase | Description |
|-----|-------|-------------|
| `loss/mse` | train | Photometric MSE between rendered and GT target |
| `loss/lpips` | train | LPIPS perceptual loss |
| `loss/damv2_pearson` | train | DAMV2 Pearson correlation loss (`-ρ(cv_depth, -damv2)`); **−0.7~−0.9 = ideal** (strong correct correlation); ≈0 = flat-depth collapse (not convergence) |
| `loss/total` | train | Sum of all active losses + opacity regularisation |
| `depth/pearson_view{v}` | train | Per-context-view Pearson correlation (observation only, no gradient) |
| `depth/pearson_avg` | train | Mean Pearson across all context views (observation only) |
| `val/lpips_val` | val | LPIPS on validation target views |
| `val/ssim_val` | val | SSIM on validation target views |
| `val/mse_val` | val | Pixel-wise MSE on validation target views |
| `val/render_vs_gt` | val | Side-by-side image: GT (left) vs render (right) |

---

## Code Structure (SkySplat-specific)

```
src/
├── model/
│   ├── encoder/
│   │   └── encoder_costvolume.py     # RPC camera setup, altitude near/far,
│   │                                 # WGS84→ENU Gaussian position conversion
│   ├── decoder/
│   │   └── cuda_splatting.py         # Post-render inverse skew warp
│   └── model_wrapper.py              # Training loop: loads JSON cameras, computes
│                                     # skew_params, near/far from MSL altitude
├── geometry/
│   └── rpc.py                        # RPC model: forward projection, Jacobian
│                                     # linearization, inverse (pixel+height → lat/lon)
└── dataset/
    └── dataset_dfc2019.py            # Dataset loader with RPC metadata and ENU origin
```

---

## Acknowledgements

SkySplat builds on [MVSplat](https://github.com/donydchen/mvsplat) (Chen et al., ECCV 2024) and incorporates backbone weights from [UniMatch](https://github.com/autonomousvision/unimatch). The RPC geometry, satellite-specific coordinate handling, and skew correction pipeline are original contributions.

```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and
               Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
```
