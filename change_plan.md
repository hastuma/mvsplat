# Architectural Change: Drop Target Views (training only)

## Goal

Train on **3 context views only** — no target view. Photometric losses are computed on the context views themselves: render Gaussians back at the context cameras and supervise against the input context images. This is the correct formulation for the satellite reconstruction setting.

**Validation and test stages keep the held-out target view** — `val/lpips_val`, `val/ssim_val`, `val/mse_val` continue to measure novel-view quality on a held-out frame. Only the training-time `target` is dropped.

The change is mostly **deletion + reference renames**. The encoder, cost volume, RPC math, ENU setup, and skew correction are already context-only — none of them touch `batch["target"]`.

---

## What `batch["target"]` does today

The target view's only role is loss supervision: the decoder renders Gaussians at the target camera, and `LossMse` / `LossLpips` compare the render against `target["image"]`. The encoder never sees the target. So removing target from training = removing one render call against one extra view, then redirecting loss to context.

The existing `vis_context` block at [src/model/model_wrapper.py:484-549](src/model/model_wrapper.py#L484-L549) already renders at context cameras for monitoring — that is the template for the new main render path.

---

## File-by-file changes

### 1. [src/dataset/dataset_dfc2019.py](src/dataset/dataset_dfc2019.py)

Drop target **only when `self.stage == "train"`**. Val and test stages keep the existing 3-context + 1-target structure.

**Online crop path (`_getitem_online`, [L300-L478](src/dataset/dataset_dfc2019.py#L300-L478))** — only used when `stage == "train"`:
- Sample 3 images instead of 4 — `random.sample(range(len(images)), min(3, len(images)))` at [L310](src/dataset/dataset_dfc2019.py#L310).
- Set `num_context = 3`, drop `num_target` ([L399-L401](src/dataset/dataset_dfc2019.py#L399-L401)).
- Remove the `tgt_views = _make_views(...)` call ([L438](src/dataset/dataset_dfc2019.py#L438)) and the `"target": {...}` block in the return dict ([L462-L476](src/dataset/dataset_dfc2019.py#L462-L476)).

**Pre-cropped path (`_getitem_precropped`, [L484-L650](src/dataset/dataset_dfc2019.py#L484-L650))** — used in train (when no raw_scenes_dir), val, and test:
- For `self.stage == "train"`: require `len(files) >= 3`, build only the 3-context portion of the return dict, omit the `"target": {...}` block.
- For `self.stage in ("val", "test")`: leave the existing 4-view (3+1) behavior intact.

### 2. [src/model/model_wrapper.py](src/model/model_wrapper.py)

In `training_step` only:
- Replace `batch["target"]` access at [L414](src/model/model_wrapper.py#L414) with `batch["context"]`.
- Delete the entire `if "rpc" in batch["context"] and "rpc" in batch["target"]:` block at [L418-L431](src/model/model_wrapper.py#L418-L431) that loads target cameras.
- Keep the context camera load at [L432-L439](src/model/model_wrapper.py#L432-L439); capture its 5th return value as `distance_ctx_flat` and rearrange to `dist_ctx` of shape `[B, V]`.
- Compute `render_near` / `render_far` from context using the **vis_context wide-sweep formula** (mirrors [L529-L530](src/model/model_wrapper.py#L529-L530) exactly):
  ```python
  render_near = (dist_ctx - 50000.0).clamp(min=1.0)
  render_far  = (dist_ctx + 50000.0).clamp(min=render_near + 1.0)
  ```
  (Wide ±50000m because satellite-scale camera distances are large; the tight ±20m used for the original target render leaves Gaussians outside the rasterizer frustum.)
- Render once at context cameras:
  ```python
  output = self.decoder.forward(
      gaussians,
      batch["context"]["extrinsics"], batch["context"]["intrinsics"],
      render_near, render_far, (h, w),
      depth_mode=self.train_cfg.depth_mode,
      skew_params=batch["context"].get("skew_params"),
      crop_offsets=batch["context"].get("crop_offsets"),
  )
  ```
- `target_gt = batch["target"]["image"]` ([L468](src/model/model_wrapper.py#L468)) → `target_gt = batch["context"]["image"]`.
- PLY export at [L663](src/model/model_wrapper.py#L663): `batch["target"]["extrinsics"][b_idx, 0]` → `batch["context"]["extrinsics"][b_idx, 0]` (use context view 0 as the reference camera).
- PLY export at [L685](src/model/model_wrapper.py#L685): redirect to context — `"target_extrinsics": batch["context"]["extrinsics"][b_idx].detach().cpu()` (key kept as `target_extrinsics` for downstream-script compatibility, OR rename to `render_extrinsics` if no consumer relies on the name).
- Delete the standalone `vis_context` block at [L484-L549](src/model/model_wrapper.py#L484-L549) — its purpose is now served by the main render path.

In `validation_step` and `test_step`:
- **No changes to camera/render flow** — these still use `batch["target"]`. Existing val/lpips_val, val/ssim_val, val/mse_val measure novel-view quality on the held-out target frame.
- Skip `for loss_fn in self.losses` in val (around [L966-L969](src/model/model_wrapper.py#L966-L969)) — those loss functions now read `batch["context"]["image"]` while `output_softmax` is rendered at target cameras, so they would compare apples to oranges. Manual `val/lpips_val` etc. are sufficient.
- Delete the standalone context-render block in `test_step` ([L774-L828](src/model/model_wrapper.py#L774-L828)) — purely for parity with the simplification in training_step (it's a save_image side-effect, optional to keep). **Skip this in the smoke test** if it's not on the critical path.

### 3. Loss functions

These run during training_step (where target is gone). Switch all to context:

- [src/loss/loss_mse.py:30](src/loss/loss_mse.py#L30) — `batch["target"]["image"]` → `batch["context"]["image"]`.
- [src/loss/loss_lpips.py:43](src/loss/loss_lpips.py#L43) — same.
- [src/loss/loss_depth.py:35-36, 51](src/loss/loss_depth.py#L35-L51) — `batch["target"]["near"|"far"|"image"]` → `batch["context"][...]`.

(Loss-fn calls in val/test are skipped per §2 — so the switch only affects the train path.)

### 4. [src/dataset/types.py](src/dataset/types.py)

**No changes.** `target` field stays in `BatchedExample` / `UnbatchedExample` because val/test still populate it.

### 5. [config/experiment/dfc2019.yaml](config/experiment/dfc2019.yaml)

No changes required (the DFC2019 dataset doesn't use `view_sampler.num_target_views`). `damv2_loss_weight` is currently `1.0` (already lowered from earlier `10000.0`).

### 6. [skysplat.md](skysplat.md)

- Update "Input: $N$ satellite image patches (e.g., 3 context views + 1 target view)" → "3 context views during training; val/test additionally hold out 1 target view for novel-view metrics".
- Update §7 Loss Functions: $L_\text{RGB}$ is now computed on context views (averaged across V=3) during training, not on a held-out target.
- Add a one-line note: **near/far for the context render uses a wide `dist ± 50000m` sweep** to keep all Gaussians inside the rasterizer's frustum given satellite-scale camera distances.

### 7. Things deferred (fix on first error)

- `src/dataset/shims/{bounds,patch,augmentation,crop}_shim.py` — these read `batch["target"]`. If they're invoked on a train batch without target, they'll `KeyError`. Defer until a smoke test surfaces it.
- `src/evaluation/{evaluation_index_generator,metric_computer}.py` — only reachable via test/eval entry points; not on the train path.

---

## Things that **do not** change

- Encoder ([encoder_costvolume.py](src/model/encoder/encoder_costvolume.py)): already context-only.
- RPC geometry ([src/geometry/rpc.py](src/geometry/rpc.py)).
- Skew correction params + inverse warp ([src/model/decoder/cuda_splatting.py](src/model/decoder/cuda_splatting.py)).
- DAMV2 Pearson loss: already operates on context views.
- ENU origin / WGS84 conversion.
- View samplers (re10k path is unaffected; DFC2019 doesn't use them).
- Validation wrapper, data module, optimizer, checkpointing.
- Validation / test camera and render flow (continue to use target).

---

## Behavioural impact

- **Loss magnitude:** photometric loss is now averaged over V=3 context views instead of 1 target view. Absolute loss values won't be comparable to old runs; expect ~3× more gradient signal per step.
- **Memory:** train decoder render call now produces a `[B, 3, 3, H, W]` color tensor instead of `[B, 1, 3, H, W]`. LPIPS forward over 3 views per step. Expect ~3× train decoder + LPIPS time.
- **Validation metrics:** unchanged in meaning — `val/lpips_val`, `val/ssim_val`, `val/mse_val` still measure held-out novel-view quality.
- **Risk (accepted):** own-view rendering (Gaussians of view k rendered at view k's camera) reprojects to source pixels regardless of predicted altitude, so ~1/3 of the photometric supervision is geometrically degenerate. Cross-view terms (Gaussians of view k rendered at view j ≠ k) and the DAMV2 Pearson loss carry the geometric signal. Keep DAMV2 on.

---

## Implementation order

1. Update `dataset_dfc2019.py` (drop target only in `train` stage).
2. Update `model_wrapper.py` `training_step` (cameras, render, loss path, PLY refs, delete vis_context block).
3. Update `model_wrapper.py` `validation_step` (skip `self.losses` loop).
4. Update loss files (mse, lpips, depth → context).
5. Update skysplat.md.
6. Smoke test: one training step on the overfit scene `JAX_004`.
