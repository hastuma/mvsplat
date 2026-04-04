# SkySplat: Feed-Forward 3D Gaussian Splatting from Satellite Imagery

**SkySplat** is a research project that adapts [MVSplat](https://github.com/donydchen/mvsplat) (ECCV 2024 Oral) for satellite remote sensing. Instead of ground-level perspective images, SkySplat takes **satellite image patches with RPC (Rational Polynomial Coefficients) metadata** as input and reconstructs geo-aligned 3D Gaussian Splatting (3DGS) representations in a single forward pass — no per-scene optimization required.

> **Status**: Active debugging. Primary issue: rendered images contain dark holes, suspected to be caused by camera–Gaussian position misalignment in the coordinate system transformation.

---

## Key Differences from MVSplat

| Aspect | MVSplat | SkySplat |
|---|---|---|
| Input | Ground-level RGB images | Satellite image patches |
| Camera model | Perspective pinhole | RPC (Rational Polynomial Coefficients) |
| Camera parameters | Learned / dataset-provided | Computed already , just read from the dataset |
| Coordinate system | Camera-centric | WGS84 → ENU local tangent plane |
| Depth axis | Distance from camera | Altitude (MSL) converted to camera distance |
| Cost volume | Epipolar plane sweep | Altitude-range plane sweep via RPC warping |

---

## Pipeline

### 1. Input & Preprocessing

- **Input**: $N$ satellite image patches (e.g., 3 context views + 1 target view), each cropped to a fixed resolution (e.g., 256×256).
- **Metadata**: Each patch is accompanied by its **RPC file** — 80 coefficients encoding the non-linear mapping from (lon, lat, height) → (col, row) in the original full-resolution image.
- **Patch Offset**: The pixel offset of the patch within the original full image is recorded to enable principal-point correction.
- **ENU Origin**: A shared geographic reference point (lat₀, lon₀, h₀) is assigned per scene, used as the origin of the local ENU coordinate system.

### 2. RPC Geometry & Virtual Camera Construction

The RPC model is highly non-linear and cannot be used directly by the standard 3DGS rasterizer. SkySplat approximates the RPC as a **linear pinhole camera** over the local patch region:

1. **Jacobian Linearization**: The RPC projection function is linearized around the scene center altitude to derive an approximate intrinsic matrix $K$ and extrinsic $[R|t]$.
2. **Principal Point Correction**: The projection residual between the true RPC-projected scene center and the patch image center is used to correct $(c_x, c_y)$ in $K$, ensuring the scene center reprojects exactly to the image center (128, 128).
3. **Asymmetric Frustum**: Because the principal point may lie far outside the patch image bounds, an **asymmetric projection matrix** is constructed to handle off-center rendering correctly.
4. **Virtual Camera Distance**: The approximate distance from the camera to the ground plane is derived from the RPC GSD (Ground Sampling Distance) and focal length, used to define per-view near/far bounds.

The resulting $(K, [R|t])$ pairs replace the standard intrinsics/extrinsics in the MVSplat pipeline.

### 3. Cost Volume Construction & Height Estimation

- **Altitude-Range Plane Sweep**: Instead of sweeping over disparity or depth, SkySplat sweeps over a predefined **altitude range** (e.g., HEIGHT_OFF ± 20 m MSL).
- **RPC Warping**: For each altitude hypothesis $H_k$, source patches are warped into the reference view using the RPC inverse projection, constructing a **4D Cost Volume** of shape `[B, V, D, H, W]`.
- **Height Probability**: A UNet-based CNN processes the cost volume and predicts a per-pixel probability distribution over altitude candidates.
- **Best Height $H^*$**: The argmax (or soft-argmax) of the distribution gives the estimated altitude per pixel.

### 4. 3D Gaussian Parameter Prediction

- **Anchor Position**: Using $H^*$ from Step 3, the RPC inverse transform maps each pixel $(u, v)$ + altitude $H^*$ to a geographic coordinate (lon, lat, H) in WGS84. This serves as the **fixed position anchor** for each Gaussian.
- **Residual Learning (Encoder)**: The MVSplat encoder backbone (UniMatch-pretrained multi-view Transformer) predicts:
  - **$\Delta \text{Pos}$** — sub-pixel position residuals to refine alignment.
  - **Scale, Opacity** — geometric footprint and visibility.
  - **SH Coefficients** — view-dependent appearance encoded as Spherical Harmonics.
- **XY Freeze**: In RPC mode, the $(x, y)$ offset within each pixel is frozen to 0.5 (pixel center), so only the RPC inverse defines the lateral position. Only the depth/altitude axis is learned.

### 5. Coordinate Realignment: WGS84 → ENU

After obtaining (lat, lon, H) for each Gaussian, positions are transformed into the **ENU (East-North-Up)** local tangent plane coordinate system to avoid floating-point precision loss at large geographic coordinates:

1. **ECEF Conversion**: Both the Gaussian positions and the ENU origin are converted from geodetic (lat, lon, h) to ECEF $(X, Y, Z)$ using the WGS84 ellipsoid model.
2. **ENU Rotation**: The ECEF difference vector is rotated using the reference point's rotation matrix into $(E, N, U)$ coordinates.
3. **Scene Centering**: All Gaussian positions are expressed relative to the scene center, with $Z$ pointing up (perpendicular to the local horizontal plane).

The target camera parameters (extrinsics) are computed in the same ENU frame, ensuring Gaussians and cameras share a consistent coordinate system.

### 6. Differentiable Rendering & Optimization

- **3DGS Rasterizer**: The geo-aligned Gaussians are rendered using the standard differentiable 3DGS rasterizer with the ENU-frame camera parameters.
- **Loss Functions**:
  - **$L_\text{RGB}$**: Photometric reconstruction loss between rendered and target images.
  - **$L_\text{Smooth}$**: Regularization on $\Delta \text{Pos}$ to prevent geometry collapse — penalizes large deviations from the RPC anchor position, enforcing geometric consistency.

---

## Current Known Issues

- **Dark Holes in Rendered Images**: The primary debugging target. Hypothesized cause: the virtual pinhole cameras computed from RPC linearization are not perfectly aligned with the coordinate frame used for the Gaussian positions after WGS84 → ENU conversion. Even small rotational or translational mismatches between the camera frustum and Gaussian cloud will cause the rasterizer to miss surfaces, producing black/dark regions.

---

## Code Structure (SkySplat-specific)

```
src/
├── model/
│   └── encoder/
│       └── encoder_costvolume.py     # RPC camera construction, altitude-based near/far,
│                                     # WGS84→ENU Gaussian position conversion
├── geometry/
│   └── rpc.py                        # RPC model: forward projection, Jacobian
│                                     # linearization, inverse (pixel+height → lat/lon)
└── dataset/
    └── ...                           # Dataset loader with RPC metadata and ENU origin
```

---

## Acknowledgements

SkySplat builds on [MVSplat](https://github.com/donydchen/mvsplat) (Chen et al., ECCV 2024) and incorporates backbone weights from [UniMatch](https://github.com/autonomousvision/unimatch). The RPC geometry and satellite-specific coordinate handling are original contributions.

```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and
               Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
```
