"""
完整的视锥体检查（Frustum Culling Check）
应该替换 model_wrapper.py L233-247 的代码
"""
import torch

def check_gaussians_in_frustum_correct(
    gaussians_means,  # [B, N, 3] in world coords
    extrinsics,       # [B, V, 4, 4] camera-to-world
    intrinsics,       # [B, V, 3, 3]
    near,             # [B, V]
    far,              # [B, V]
    image_shape,      # (h, w)
):
    """
    完整检查高斯球是否在视锥体内（包括深度和FOV）
    """
    b, v = extrinsics.shape[:2]
    h, w = image_shape
    
    # 展开为 [B*V]
    tgt_ext = extrinsics.view(b*v, 4, 4)
    tgt_int = intrinsics.view(b*v, 3, 3)
    tgt_near = near.view(b*v)
    tgt_far = far.view(b*v)
    
    # 展开 Gaussians
    g_means_flat = gaussians_means.view(-1, 3)  # [B*N, 3]
    
    # === 1. 转换到相机坐标系 ===
    w2c = torch.inverse(tgt_ext[0])  # 取第一个 batch
    gs_in_cam = (w2c[:3, :3] @ g_means_flat.T + w2c[:3, 3:4]).T  # [N, 3]
    
    # === 2. 深度检查 ===
    gs_depths = gs_in_cam[:, 2]  # Z in camera coords
    depth_valid = (gs_depths > tgt_near[0]) & (gs_depths < tgt_far[0])
    
    # === 3. 投影到图像平面（NDC 检查）===
    # 使用内参投影到像素坐标
    K = tgt_int[0]  # [3, 3]
    gs_proj = K @ gs_in_cam.T  # [3, N]
    gs_proj = gs_proj[:2] / gs_proj[2:3].clamp(min=1e-8)  # [2, N] 归一化
    
    # 检查是否在图像范围内
    u, v_coord = gs_proj[0], gs_proj[1]
    fov_valid = (u > 0) & (u < w) & (v_coord > 0) & (v_coord < h)
    
    # === 4. 综合判断 ===
    in_frustum = depth_valid & fov_valid & (gs_depths > 0)  # 必须在相机前方
    
    # === 5. 统计 ===
    total = len(gs_depths)
    print(f"\n📊 [CORRECT FRUSTUM CHECK] Gaussians 可见性分析:")
    print(f"   总数: {total}")
    print(f"   深度有效 (near<z<far): {depth_valid.sum().item()} ({depth_valid.sum().item()/total*100:.1f}%)")
    print(f"   FOV 有效 (在视角内): {fov_valid.sum().item()} ({fov_valid.sum().item()/total*100:.1f}%)")
    print(f"   在相机前方 (z>0): {(gs_depths>0).sum().item()} ({(gs_depths>0).sum().item()/total*100:.1f}%)")
    print(f"   ✅ 完全可见 (在视锥体内): {in_frustum.sum().item()} ({in_frustum.sum().item()/total*100:.1f}%)")
    print(f"   ❌ 在相机后面: {(gs_depths<0).sum().item()} ({(gs_depths<0).sum().item()/total*100:.1f}%)")
    
    return in_frustum

# 使用示例（在 model_wrapper.py L233 替换）:
"""
# 旧代码（不完整）
in_frustum_depth = (gs_depths > tgt_near) & (gs_depths < tgt_far)

# 新代码（完整）
in_frustum = check_gaussians_in_frustum_correct(
    gaussians.means,
    batch["target"]["extrinsics"],
    batch["target"]["intrinsics"],
    batch["target"]["near"],
    batch["target"]["far"],
    (h, w)
)
"""
