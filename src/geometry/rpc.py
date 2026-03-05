import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

class RPC:
    def __init__(self, coeffs: Tensor, device=None):
        """
        coeffs: Tensor of shape (..., 90)
        """
        if device is None:
            device = coeffs.device
        self.device = device
        
        # 0-9: Offsets and Scales
        self.line_off = coeffs[..., 0]
        self.line_scale = coeffs[..., 1]
        self.samp_off = coeffs[..., 2]
        self.samp_scale = coeffs[..., 3]
        self.lat_off = coeffs[..., 4]
        self.lat_scale = coeffs[..., 5]
        self.long_off = coeffs[..., 6]
        self.long_scale = coeffs[..., 7]
        self.height_off = coeffs[..., 8]
        self.height_scale = coeffs[..., 9]
        
        # 10-89: Polynomial Coefficients
        self.line_num_coeff = coeffs[..., 10:30]
        self.line_den_coeff = coeffs[..., 30:50]
        self.samp_num_coeff = coeffs[..., 50:70]
        self.samp_den_coeff = coeffs[..., 70:90]

    def _polynomial(self, p, l, h, coeffs):
        # p: lat_norm, l: lon_norm, h: height_norm
        spatial_dims = p.ndim - (coeffs.ndim - 1)
        c = []
        for i in range(20):
            coef = coeffs[..., i]
            if spatial_dims > 0:
                coef = coef.view(*coef.shape, *([1] * spatial_dims))
            c.append(coef)

        # GDAL / Standard RPC 20-term Polynomial:
        # 1, l, p, h, l*p, l*h, p*h, l^2, p^2, h^2, p*l*h, l^3, l*p^2, l*h^2, l^2*p, p^3, p*h^2, l^2*h, p^2*h, h^3
        result = (c[0] + c[1]*l + c[2]*p + c[3]*h +
                  c[4]*l*p + c[5]*l*h + c[6]*p*h +
                  c[7]*l*l + c[8]*p*p + c[9]*h*h +
                  c[10]*p*l*h + c[11]*l*l*l + c[12]*l*p*p + c[13]*l*h*h + 
                  c[14]*l*l*p + c[15]*p*p*p + c[16]*p*h*h + 
                  c[17]*l*l*h + c[18]*p*p*h + c[19]*h*h*h)
        return result

    def forward(self, lat: Tensor, lon: Tensor, height: Tensor) -> Tuple[Tensor, Tensor]:
        s = [1] * (lat.ndim - self.lat_off.ndim)
        P = (lat - self.lat_off.view(*self.lat_off.shape, *s)) / self.lat_scale.view(*self.lat_scale.shape, *s)
        L = (lon - self.long_off.view(*self.long_off.shape, *s)) / self.long_scale.view(*self.long_scale.shape, *s)
        H = (height - self.height_off.view(*self.height_off.shape, *s)) / self.height_scale.view(*self.height_scale.shape, *s)

        num_l = self._polynomial(P, L, H, self.line_num_coeff)
        den_l = self._polynomial(P, L, H, self.line_den_coeff)
        num_s = self._polynomial(P, L, H, self.samp_num_coeff)
        den_s = self._polynomial(P, L, H, self.samp_den_coeff)

        row = (num_l / den_l) * self.line_scale.view(*self.line_scale.shape, *s) + self.line_off.view(*self.line_off.shape, *s)
        col = (num_s / den_s) * self.samp_scale.view(*self.samp_scale.shape, *s) + self.samp_off.view(*self.samp_off.shape, *s)
        return row, col

    def inverse(self, row: Tensor, col: Tensor, height: Tensor, initial_guess: Tuple[Tensor, Tensor] = None, iterations=10) -> Tuple[Tensor, Tensor]:
        spatial_dims = row.ndim - self.line_off.ndim
        s = [1] * spatial_dims
        
        orig_dtype = row.dtype
        Rn = ((row - self.line_off.view(*self.line_off.shape, *s)) / self.line_scale.view(*self.line_scale.shape, *s)).to(torch.float64)
        Cn = ((col - self.samp_off.view(*self.samp_off.shape, *s)) / self.samp_scale.view(*self.samp_scale.shape, *s)).to(torch.float64)
        H = ((height - self.height_off.view(*self.height_off.shape, *s)) / self.height_scale.view(*self.height_scale.shape, *s)).to(torch.float64)

        if initial_guess is None:
            P = torch.zeros_like(Rn, dtype=torch.float64)
            L = torch.zeros_like(Cn, dtype=torch.float64)
        else:
            lat_g, lon_g = initial_guess
            P = ((lat_g - self.lat_off.view(*self.lat_off.shape, *s)) / self.lat_scale.view(*self.lat_scale.shape, *s)).to(torch.float64)
            L = ((lon_g - self.long_off.view(*self.long_off.shape, *s)) / self.long_scale.view(*self.long_scale.shape, *s)).to(torch.float64)

        lnc, ldc = self.line_num_coeff.to(torch.float64), self.line_den_coeff.to(torch.float64)
        snc, sdc = self.samp_num_coeff.to(torch.float64), self.samp_den_coeff.to(torch.float64)

        for _ in range(iterations):
            with torch.enable_grad():
                P_l = P.detach().requires_grad_(True)
                L_l = L.detach().requires_grad_(True)
                
                r_hat = self._polynomial(P_l, L_l, H, lnc) / self._polynomial(P_l, L_l, H, ldc)
                c_hat = self._polynomial(P_l, L_l, H, snc) / self._polynomial(P_l, L_l, H, sdc)
                
                row_grads = torch.autograd.grad(r_hat.sum(), [P_l, L_l])
                col_grads = torch.autograd.grad(c_hat.sum(), [P_l, L_l])
                
                J = torch.stack([
                    torch.stack([row_grads[0], row_grads[1]], dim=-1),
                    torch.stack([col_grads[0], col_grads[1]], dim=-1)
                ], dim=-2)
                
                res = torch.stack([Rn - r_hat.detach(), Cn - c_hat.detach()], dim=-1)
                dX = torch.linalg.solve(J, res.unsqueeze(-1)).squeeze(-1)
                P = P + dX[..., 0]
                L = L + dX[..., 1]

        lat = P * self.lat_scale.view(*self.lat_scale.shape, *s).to(torch.float64) + self.lat_off.view(*self.lat_off.shape, *s).to(torch.float64)
        lon = L * self.long_scale.view(*self.long_scale.shape, *s).to(torch.float64) + self.long_off.view(*self.long_off.shape, *s).to(torch.float64)
        return lat.to(orig_dtype), lon.to(orig_dtype)

    def get_pinhole_approximation(self, u: Tensor, v: Tensor, h: Tensor, image_size: Tuple[int, int] = (256, 256)) -> Tuple[Tensor, Tensor]:
        """
        計算 RPC 在給定點的局部針孔相機近似。
        
        Args:
            u, v: 像素座標 (row, col)
            h: 高度 (MSL)
            image_size: (height, width) 用於設定主點
        
        Returns:
            K: 3x3 內參矩陣
            R_c2w: 3x3 旋轉矩陣 (camera-to-world)
                   相機座標系定義：
                   - X 軸：指向圖像右方
                   - Y 軸：指向圖像下方  
                   - Z 軸：指向場景（從相機看出去的方向）
        """
        img_h, img_w = image_size
        
        with torch.no_grad():
            lat, lon = self.inverse(u, v, h)
        
        with torch.enable_grad():
            r_earth = 6378137.0
            deg_to_rad = 3.1415926535 / 180.0
            lat_v = lat.detach().clone().requires_grad_(True)
            lon_v = lon.detach().clone().requires_grad_(True)
            h_v = h.detach().clone().requires_grad_(True)
            row, col = self.forward(lat_v, lon_v, h_v)
            
            du_dlat = torch.autograd.grad(row.sum(), lat_v, retain_graph=True)[0]
            du_dlon = torch.autograd.grad(row.sum(), lon_v, retain_graph=True)[0]
            dv_dlat = torch.autograd.grad(col.sum(), lat_v, retain_graph=True)[0]
            dv_dlon = torch.autograd.grad(col.sum(), lon_v, retain_graph=True)[0]
            
            cos_lat = torch.cos(lat * deg_to_rad)
            
        # Jacobian: d(pixel) / d(geo)
        J = torch.stack([
            torch.stack([du_dlon, du_dlat], dim=-1),
            torch.stack([dv_dlon, dv_dlat], dim=-1)
        ], dim=-2)
        J_inv = torch.linalg.inv(J + torch.eye(2, device=self.device).unsqueeze(0) * 1e-12)
        
        # === 光線追蹤計算真實 Z 軸 ===
        # 在同一像素上計算兩個不同高度的 3D 座標，連線得到相機視線方向
        h_ground = h  # 地面高度
        h_sky = h + 100.0  # 天空高度（+100m）
        
        # 反投影得到兩個 lat/lon
        with torch.no_grad():
            lat_ground, lon_ground = self.inverse(u, v, h_ground)
            lat_sky, lon_sky = self.inverse(u, v, h_sky)
        
        # 轉換為 ENU 座標（相對於地面點）
        x_ground = torch.zeros_like(lat_ground)
        y_ground = torch.zeros_like(lat_ground)
        z_ground = torch.zeros_like(lat_ground)
        
        x_sky = (lon_sky - lon_ground) * deg_to_rad * r_earth * torch.cos(lat_ground * deg_to_rad)
        y_sky = (lat_sky - lat_ground) * deg_to_rad * r_earth
        z_sky = torch.full_like(lat_sky, 100.0)  # 高度差
        
        # 視線方向（從地面指向天空）= 相機 Z 軸的負方向
        ray_dir = torch.stack([x_sky - x_ground, y_sky - y_ground, z_sky - z_ground], dim=-1)
        ray_dir_normalized = ray_dir / ray_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # 相機 Z 軸應該指向地面（視線的反方向）
        cam_z = -ray_dir_normalized
        
        # 計算 c_right, c_down（水平分量，用於計算 GSD）
        dx_du = J_inv[..., 0, 0] * deg_to_rad * r_earth * cos_lat
        dy_du = J_inv[..., 1, 0] * deg_to_rad * r_earth
        c_right_horizontal = torch.stack([dx_du, dy_du, torch.zeros_like(dx_du)], dim=-1)
        
        dx_dv = J_inv[..., 0, 1] * deg_to_rad * r_earth * cos_lat
        dy_dv = J_inv[..., 1, 1] * deg_to_rad * r_earth
        c_down_horizontal = torch.stack([dx_dv, dy_dv, torch.zeros_like(dx_dv)], dim=-1)
        
        # 焦距 = 1 / GSD_horizontal（使用水平分量計算）
        fx = 1.0 / c_right_horizontal.norm(dim=-1).clamp(min=1e-8)
        fy = 1.0 / c_down_horizontal.norm(dim=-1).clamp(min=1e-8)
        
        # 建構相機座標系（camera-to-world）
        # 相機 X 軸：指向右（投影到垂直於 Z 的平面）
        c_right_proj = c_right_horizontal - (c_right_horizontal * cam_z).sum(dim=-1, keepdim=True) * cam_z
        cam_x = c_right_proj / c_right_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # 相機 Y 軸：cross(Z, X) 確保右手座標系
        cam_y = torch.cross(cam_z, cam_x, dim=-1)
        cam_y = cam_y / cam_y.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # print(f"u {u} \n v {v} \n h {h} \n image_size {image_size}")
        # print(f"cam_z {cam_z} \n cam_x {cam_x} \n cam_y {cam_y} \n fx {fx} \n fy {fy}")
        # exit()

        # === DEBUG: 檢查 Z 軸是否有傾角 ===
        z_vertical = torch.tensor([0.0, 0.0, -1.0], device=cam_z.device)
        cos_angle = torch.clamp((cam_z * z_vertical).sum(dim=-1), -1, 1)
        angle_deg = torch.acos(cos_angle) * 180.0 / 3.14159265
        z_horizontal = torch.sqrt(cam_z[..., 0]**2 + cam_z[..., 1]**2)

        # R_c2w: 每一列是相機座標軸在世界座標中的方向
        R_c2w = torch.stack([cam_x, cam_y, cam_z], dim=-1)
        
        # 內參矩陣
        K = torch.eye(3, device=self.device).repeat(*cam_x.shape[:-1], 1, 1)
        K[..., 0, 0], K[..., 1, 1] = fx, fy
        K[..., 0, 2], K[..., 1, 2] = img_w / 2.0, img_h / 2.0
        
        return K.to(dtype=torch.float32), R_c2w.to(dtype=torch.float32)
    def compute_camera_geometry(self, h: int, w: int, lat_ref_global: Tensor = None, lon_ref_global: Tensor = None):
        """
        從 RPC 近似針孔相機的內外參數。完全由 RPC 數據驅動，物理單位正確。
        
        Args:
            h, w: 圖像尺寸（像素）
            lat_ref_global, lon_ref_global: ENU 座標系原點的經緯度
        
        Returns:
            K: [B, 3, 3] 內參矩陣
            c2w: [B, 4, 4] camera-to-world 外參矩陣（含真實傾角，非強制 Nadir）
            distance: [B] 虛擬相機到場景參考面的高度（米），供 near/far 計算使用
        
        推導方式（物理正確）：
            1. 用 RPC Jacobian 在 HEIGHT_OFF 計算每像素對應地面距離（GSD_rpc）
            2. H = (w/2) × GSD_rpc：讓相機以 ~90° 水平視角覆蓋圖像
               單位：像素 × (米/像素) = 米 ✓
            3. f = H / GSD_rpc = w/2：焦距（像素單位）
               單位：米 / (米/像素) = 像素 ✓
            4. 相機 Z 軸通過光線追踪計算真實朝向（非強制垂直）
        座標系定義：
            - ENU 世界座標系：X=東, Y=北, Z=上
            - 相機位於 HEIGHT_OFF + H 高度，Z 軸由光線追踪決定（可能有傾角）
        """
        device = self.device
        dtype = self.line_off.dtype
        B = self.line_off.shape[0]
        
        u_center = torch.full((B,), w / 2.0, device=device, dtype=dtype)
        v_center = torch.full((B,), h / 2.0, device=device, dtype=dtype)
        h_mean = self.height_off  # RPC 參考高度 (MSL)
        
        # Step 1: 從 RPC Jacobian 取得旋轉矩陣 R 及局部尺度
        # K_jac[..., 0, 0] = 1/GSD_rpc_x (pixels/meter at HEIGHT_OFF)
        K_jac, R_c2w = self.get_pinhole_approximation(u_center, v_center, h_mean, image_size=(h, w))
        
        # Step 2: 從 Jacobian 推導 GSD（米/像素）← 這是 RPC 在參考高度的真實解析度
        gsd_rpc_x = 1.0 / K_jac[..., 0, 0].clamp(min=1e-6)  # [B], meters per pixel
        gsd_rpc_y = 1.0 / K_jac[..., 1, 1].clamp(min=1e-6)  # [B], meters per pixel
        gsd_rpc = (gsd_rpc_x + gsd_rpc_y) * 0.5             # 平均 GSD [B]
        
        # Step 3: 虛擬相機高度（物理正確公式）
        # H = (w/2) × GSD：讓相機剛好以 ~90° 水平視角覆蓋圖像
        # 單位檢查：像素 × (米/像素) = 米 ✓
        distance = (w / 2.0) * gsd_rpc  # [B], meters
        # distance = self.height_scale * 1.5  # [B], meters
        
        # Step 4: 焦距（物理正確公式）
        # f = H / GSD 或直接使用 Jacobian 推導的焦距
        # 單位檢查：米 / (米/像素) = 像素 ✓
        focal = distance / gsd_rpc  # [B], pixels; 等價於 w/2
        
        # === DEBUG: 驗證單位正確性 ===
        print(f"\n[RPC GEOMETRY DEBUG]")
        print(f"  GSD (from Jacobian): {gsd_rpc[0].item():.4f} m/px")
        print(f"  Camera Height H = (w/2) × GSD = {w/2:.0f} × {gsd_rpc[0].item():.4f} = {distance[0].item():.2f} m")
        print(f"  Focal f = H / GSD = {distance[0].item():.2f} / {gsd_rpc[0].item():.4f} = {focal[0].item():.2f} px")
        print(f"  Expected: f ≈ w/2 = {w/2:.0f} px ✓" if abs(focal[0].item() - w/2) < 1 else f"  WARNING: focal != w/2")
        print(f"  height_scale: {self.height_scale[0].item():.1f} m (RPC norm param)")
        # input("  Press Enter to continue...")  # 暫停以檢查輸出
        # 或者直接用 Jacobian: focal = K_jac.diagonal()[..., :2].mean()
        
        # 建立 K
        K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        K[..., 0, 0] = focal
        K[..., 1, 1] = focal
        K[..., 0, 2] = w / 2.0
        K[..., 1, 2] = h / 2.0
        
        # Step 5: 計算相機在 ENU 座標系中的位置
        with torch.no_grad():
            lat_c, lon_c = self.inverse(u_center, v_center, h_mean)
        
        lat_ref = lat_c  if lat_ref_global is None else lat_ref_global
        lon_ref = lon_c  if lon_ref_global is None else lon_ref_global
        r_earth, rad = 6378137.0, 3.1415926535 / 180.0
        print("lat_ref:", lat_ref, "lon_ref:", lon_ref)
        # exit()
        # ENU 座標（相對於 ENU 原點）
        x_c = (lon_c - lon_ref) * rad * r_earth * torch.cos(lat_ref * rad)  # 東
        y_c = (lat_c  - lat_ref) * rad * r_earth                             # 北
        z_c = h_mean + distance                                               # 上（MSL）
        
        # 建構 4x4 c2w 矩陣
        c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        c2w[:, :3, :3] = R_c2w
        c2w[:, :3, 3] = torch.stack([x_c, y_c, z_c.to(dtype)], dim=-1)
        print(f"camera position info - lat: {lat_c[0].item():.6f}, lon: {lon_c[0].item():.6f}, x: {x_c[0].item():.2f} m, y: {y_c[0].item():.2f} m, z: {z_c[0].item():.2f} m (MSL)")
        # input()
        return K.to(dtype=torch.float32), c2w.to(dtype=torch.float32), distance.to(dtype=torch.float32)