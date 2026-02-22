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
        
        # 計算每移動 1 像素對應世界座標的變化（米）
        # c_right: 圖像 u 方向對應的世界座標變化
        dx_du = J_inv[..., 0, 0] * deg_to_rad * r_earth * cos_lat
        dy_du = J_inv[..., 1, 0] * deg_to_rad * r_earth
        c_right = torch.stack([dx_du, dy_du, torch.zeros_like(dx_du)], dim=-1)
        
        # c_down: 圖像 v 方向對應的世界座標變化
        dx_dv = J_inv[..., 0, 1] * deg_to_rad * r_earth * cos_lat
        dy_dv = J_inv[..., 1, 1] * deg_to_rad * r_earth
        c_down = torch.stack([dx_dv, dy_dv, torch.zeros_like(dx_dv)], dim=-1)
        
        # 焦距 = 1 / GSD (Ground Sample Distance)
        fx = 1.0 / c_right.norm(dim=-1).clamp(min=1e-8)
        fy = 1.0 / c_down.norm(dim=-1).clamp(min=1e-8)
        
        # 建構相機座標系（camera-to-world）
        # 相機 X 軸：指向右（與 c_right 對齊）
        cam_x = c_right / c_right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # 相機 Z 軸：cross(right, down) 會給出「向上」的向量
        # 但我們需要相機看「向下」（朝向地面），所以取負號
        cam_z_up = torch.cross(c_right, c_down, dim=-1)
        cam_z = -cam_z_up / cam_z_up.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # 翻轉使 Z 朝下
        
        # 相機 Y 軸：cross(Z, X) 確保右手座標系
        cam_y = torch.cross(cam_z, cam_x, dim=-1)
        cam_y = cam_y / cam_y.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # R_c2w: 每一列是相機座標軸在世界座標中的方向
        R_c2w = torch.stack([cam_x, cam_y, cam_z], dim=-1)
        
        # 內參矩陣
        K = torch.eye(3, device=self.device).repeat(*cam_x.shape[:-1], 1, 1)
        K[..., 0, 0], K[..., 1, 1] = fx, fy
        K[..., 0, 2], K[..., 1, 2] = img_w / 2.0, img_h / 2.0
        
        return K.to(dtype=torch.float32), R_c2w.to(dtype=torch.float32)
    def compute_camera_geometry(self, h: int, w: int, lat_ref_global: Tensor = None, lon_ref_global: Tensor = None,
                                target_half_fov_deg: float = 30.0, min_distance: float = 40.0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        從 RPC 模型近似出針孔相機的內參與外參，相機高度由 RPC Jacobian
        推導出的真實 GSD 決定，不再使用硬編碼的固定值。

        Args:
            h, w: 圖像尺寸（像素）
            lat_ref_global, lon_ref_global: ENU 座標系原點的經緯度
            target_half_fov_deg: 虛擬相機的目標半視角（度），用於決定
                相機高度 distance = (image_half_footprint) / tan(target_half_fov)
                預設 30°（等效 60° 全視角），適合 cost volume 近/遠面比例。
            min_distance: distance 的最小值（米），確保 near > 0。

        Returns:
            K:        [B, 3, 3]  內參矩陣（焦距由 GSD 推算）
            c2w:      [B, 4, 4]  camera-to-world 外參矩陣
            distance: [B]        每個相機的虛擬相機高度（米，相對 HEIGHT_OFF）

        座標系定義：
            - ENU 世界座標系：X=東, Y=北, Z=上
            - 相機位於 HEIGHT_OFF + distance 高度
            - 相機 Z 軸指向地面（向下看）
        """
        import math
        device = self.device
        dtype = self.line_off.dtype
        B = self.line_off.shape[0]

        u_center = torch.full((B,), w / 2.0, device=device, dtype=dtype)
        v_center = torch.full((B,), h / 2.0, device=device, dtype=dtype)
        h_mean = self.height_off  # RPC 參考高度 (MSL)

        # Step 1: 從 RPC Jacobian 獲取局部相機方向 + 真實 GSD
        # get_pinhole_approximation 返回的 K[0,0] = 1/GSD_x, K[1,1] = 1/GSD_y
        # 其中 GSD_x, GSD_y 單位為 meters/pixel
        K, R_c2w = self.get_pinhole_approximation(u_center, v_center, h_mean, image_size=(h, w))
        K = K.clone()

        # Step 2: 從 K 解回真實 GSD（米/像素）
        # 注意：get_pinhole_approximation 中 fx = 1 / c_right.norm()，即 GSD 的倒數
        gsd_x = (1.0 / K[..., 0, 0].clamp(min=1e-8)).float()  # [B] meters/pixel
        gsd_y = (1.0 / K[..., 1, 1].clamp(min=1e-8)).float()  # [B] meters/pixel

        # Step 3: 由目標半視角計算 distance
        # tan(half_fov) = half_footprint / distance
        # half_footprint_x = (w/2) * GSD_x
        # → distance_x = (w/2 * GSD_x) / tan(half_fov)
        tan_half_fov = math.tan(math.radians(target_half_fov_deg))
        D_x = (w / 2.0) * gsd_x / tan_half_fov  # [B]
        D_y = (h / 2.0) * gsd_y / tan_half_fov  # [B]
        distance = ((D_x + D_y) / 2.0).clamp(min=float(min_distance))  # [B]

        # Step 4: 用推算出的 distance 重新計算焦距（focal = distance / GSD）
        focal_x = distance / gsd_x  # [B]
        focal_y = distance / gsd_y  # [B]
        K[..., 0, 0] = focal_x
        K[..., 1, 1] = focal_y
        K[..., 0, 2] = w / 2.0
        K[..., 1, 2] = h / 2.0

        # Step 5: 計算相機在 ENU 座標系中的位置
        with torch.no_grad():
            lat_c, lon_c = self.inverse(u_center, v_center, h_mean)

        lat_ref, lon_ref = (lat_c, lon_c) if lat_ref_global is None else (lat_ref_global, lon_ref_global)
        r_earth, rad = 6378137.0, 3.1415926535 / 180.0

        # ENU 座標（相對於參考點）
        x_c = (lon_c - lon_ref) * rad * r_earth * torch.cos(lat_ref * rad)  # 東 [B]
        y_c = (lat_c - lat_ref) * rad * r_earth                              # 北 [B]
        z_c = h_mean.float() + distance                                       # 上 [B]（相機在參考面上方 distance 米）

        # Step 6: 建構 4x4 c2w 矩陣
        c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        c2w[:, :3, :3] = R_c2w
        c2w[:, :3, 3] = torch.stack([x_c, y_c, z_c], dim=-1)

        return K.to(dtype=torch.float32), c2w.to(dtype=torch.float32), distance.to(dtype=torch.float32)