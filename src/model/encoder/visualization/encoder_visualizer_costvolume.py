from pathlib import Path
from random import randrange
from typing import Optional

import numpy as np
import torch
import wandb
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float
from torch import Tensor

from ....dataset.types import BatchedViews
from ....misc.heterogeneous_pairings import generate_heterogeneous_index
from ....visualization.annotation import add_label
from ....visualization.color_map import apply_color_map, apply_color_map_to_image
from ....visualization.colors import get_distinct_color
from ....visualization.drawing.lines import draw_lines
from ....visualization.drawing.points import draw_points
from ....visualization.layout import add_border, hcat, vcat
from ...ply_export import export_ply
from ..encoder_costvolume import EncoderCostVolume
# from ..epipolar.epipolar_sampler import EpipolarSampling
from .encoder_visualizer import EncoderVisualizer
from .encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)


class EncoderVisualizerCostVolume(
    EncoderVisualizer[EncoderVisualizerCostVolumeCfg, EncoderCostVolume]
):
    def visualize(
        self,
        context: BatchedViews,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        visualization_dump = {}

        self.encoder.forward(
            context,
            global_step,
            visualization_dump=visualization_dump,
            deterministic=True,
        )

        # Return ONLY the concatenated image the user requested
        return {
            "depth_comparison": self.visualize_depth(
                context,
                visualization_dump["depth"],
                visualization_dump.get("features", None),
            ),
        }

    def visualize_attention(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        attention: Float[Tensor, "layer bvr head 1 sample"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize attention in the sample view.
        attention = rearrange(
            attention, "l (b v r) hd () s -> l b v r hd s", b=b, v=v, r=r
        )
        attention = attention[:, rb, rv, rr, :, :]
        num_layers, _, hd, _ = attention.shape

        vis = []
        for il in range(num_layers):
            vis_layer = []
            for ihd in range(hd):
                # Create colors according to attention.
                color = [get_distinct_color(i) for i, _ in enumerate(rr)]
                color = torch.tensor(color, device=attention.device)
                color = rearrange(color, "r c -> r () c")
                attn = rearrange(attention[il, :, ihd], "r s -> r s ()")
                color = rearrange(attn * color, "r s c -> (r s ) c")

                # Draw the alternating bucket lines.
                vis_layer_head = draw_lines(
                    context_images[rb, self.encoder.sampler.index_v[rv, rov]],
                    rearrange(
                        sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"
                    ),
                    rearrange(
                        sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"
                    ),
                    color,
                    3,
                    cap="butt",
                    x_range=(0, 1),
                    y_range=(0, 1),
                )
                vis_layer.append(vis_layer_head)
            vis.append(add_label(vcat(*vis_layer), f"Layer {il}"))
        vis = add_label(add_border(add_border(hcat(*vis)), 1, 0), "Keys & Values")
        vis = add_border(hcat(add_label(ray_view, "ray_view"), vis, align="top"))
        return vis

    # def visualize_depth(
    #     self,
    #     context: BatchedViews,
    #     multi_depth: Float[Tensor, "batch view height width surface spp"],
    def visualize_depth(
        self,
        context: BatchedViews,
        multi_depth: Float[Tensor, "batch view height width surface spp"],
        features: Optional[Float[Tensor, "batch view channel feat_height feat_width"]] = None,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        # Use simple visualization for only the first sample in batch/view
        b_idx, v_idx = 0, 0
        
        # 1. Input RGB [3, H, W]
        rgb = context["image"][b_idx, v_idx].clone()
        # Min-Max Normalization for RGB to ensure it's visible (0-1 range)
        r_min, r_max = rgb.min(), rgb.max()
        rgb = (rgb - r_min) / (r_max - r_min + 1e-6)
        h, w = rgb.shape[-2:]

        # 2. Feature Map (Color - First 3 Channels) [3, H, W]
        if features is not None:
            # Use color for features to distinguish from grayscale height map
            feat = features[b_idx, v_idx, :3].clone() # [3, Hf, Wf]
            # Resize feature map to match RGB resolution
            feat = torch.nn.functional.interpolate(
                feat.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            # Min-Max Normalization
            f_min, f_max = feat.min(), feat.max()
            feat = (feat - f_min) / (f_max - f_min + 1e-6)
        else:
            feat = torch.zeros_like(rgb)

        # 3. Predicted Height Map (Grayscale) [3, H, W]
        # Average across surfaces and spp
        depth = multi_depth[b_idx, v_idx].mean(dim=(-2, -1))
        # Min-Max Normalization (Matches debug.py logic)
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
        # Convert to 3-channel grayscale for concatenation
        height_map_bw = repeat(depth_norm, "h w -> 3 h w")

        # 4. Concatenate horizontally: RGB | Feature | Height
        combined = torch.cat([rgb, feat, height_map_bw], dim=-1)

        return combined
    
    def visualize_overlaps(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        is_monocular: Optional[Bool[Tensor, "batch view height width"]] = None,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        device = context_images.device
        b, v, _, h, w = context_images.shape
        green = torch.tensor([0.235, 0.706, 0.294], device=device)[..., None, None]
        rb = randrange(b)
        valid = sampling.valid[rb].float()
        ds = self.encoder.cfg.epipolar_transformer.downscale
        valid = repeat(
            valid,
            "v ov (h w) -> v ov c (h rh) (w rw)",
            c=3,
            h=h // ds,
            w=w // ds,
            rh=ds,
            rw=ds,
        )

        if is_monocular is not None:
            is_monocular = is_monocular[rb].float()
            is_monocular = repeat(is_monocular, "v h w -> v c h w", c=3, h=h, w=w)

        # Select context images in grid.
        context_images = context_images[rb]
        index, _ = generate_heterogeneous_index(v)
        valid = valid * (green + context_images[index]) / 2

        vis = vcat(*(hcat(im, hcat(*v)) for im, v in zip(context_images, valid)))
        vis = add_label(vis, "Context Overlaps")

        if is_monocular is not None:
            vis = hcat(vis, add_label(vcat(*is_monocular), "Monocular?"))

        return add_border(vis)

    def visualize_gaussians(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        b, v, _, h, w = context_images.shape
        rb = randrange(b)
        context_images = context_images[rb]
        opacities = repeat(
            opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=h, w=w
        )
        colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        # Color-map Gaussian covariawnces.
        det = covariances[rb].det()
        det = apply_color_map(det / det.max(), "inferno")
        det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        return add_border(
            hcat(
                add_label(box(hcat(*context_images)), "Context"),
                add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
                add_label(
                    box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
                ),
                add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
                add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
            )
        )

    def visualize_probabilities(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        pdf: Float[Tensor, "batch view ray sample"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, _, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)
        colors = [get_distinct_color(i) for i, _ in enumerate(rr)]
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            colors,
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize probabilities in the sample view.
        pdf = pdf[rb, rv, rr]
        pdf = rearrange(pdf, "r s -> r s ()")
        colors = rearrange(colors, "r c -> r () c")
        sample_view = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(pdf * colors, "r s c -> (r s) c"),
            6,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize rescaled probabilities in the sample view.
        pdf_magnified = pdf / reduce(pdf, "r s () -> r () ()", "max")
        sample_view_magnified = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(pdf_magnified * colors, "r s c -> (r s) c"),
            6,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(
                add_label(ray_view, "Rays"),
                add_label(sample_view, "Samples"),
                add_label(sample_view_magnified, "Samples (Magnified PDF)"),
            )
        )

    def visualize_epipolar_samples(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize the samples and epipolar lines in the sample view.
        # First, draw the epipolar line in black.
        sample_view = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            sampling.xy_sample_near[rb, rv, rov, rr, 0],
            sampling.xy_sample_far[rb, rv, rov, rr, -1],
            0,
            5,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Create an alternating line color for the buckets.
        color = repeat(
            torch.tensor([0, 1], device=device),
            "ab -> r (s ab) c",
            r=len(rr),
            s=(s + 1) // 2,
            c=3,
        )
        color = rearrange(color[:, :s], "r s c -> (r s) c")

        # Draw the alternating bucket lines.
        sample_view = draw_lines(
            sample_view,
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            color,
            3,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Draw the sample points.
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            [get_distinct_color(i // s) for i in range(s * len(rr))],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(add_label(ray_view, "Ray View"), add_label(sample_view, "Sample View"))
        )

    def visualize_epipolar_color_samples(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        context: BatchedViews,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        sampling = self.encoder.sampler(
            context["image"],
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
        )

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize the samples and in the sample view.
        sample_view = draw_points(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            [get_distinct_color(i // s) for i in range(s * len(rr))],
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.features[rb, rv, rov, rr], "r s c -> (r s) c"),
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(add_label(ray_view, "Ray View"), add_label(sample_view, "Sample View"))
        )
