import torch
from torch import Tensor
from typing import Tuple

class RPC:
    def __init__(self, coeffs: Tensor, device=None):
        """
        coeffs: Tensor of shape (..., 90) or similar, containing:
            LINE_OFF, LINE_SCALE, SAMP_OFF, SAMP_SCALE,
            LAT_OFF, LAT_SCALE, LONG_OFF, LONG_SCALE,
            HEIGHT_OFF, HEIGHT_SCALE,
            LINE_NUM_COEFF (20), LINE_DEN_COEFF (20),
            SAMP_NUM_COEFF (20), SAMP_DEN_COEFF (20)
        """
        if device is None:
            device = coeffs.device
        self.device = device
        
        # Parse coefficients
        # Assuming the standard order which is often:
        # 0: LINE_OFF, 1: LINE_SCALE, 2: SAMP_OFF, 3: SAMP_SCALE
        # 4: LAT_OFF, 5: LAT_SCALE, 6: LONG_OFF, 7: LONG_SCALE
        # 8: HEIGHT_OFF, 9: HEIGHT_SCALE
        # 10-29: LINE_NUM_COEFF (20)
        # 30-49: LINE_DEN_COEFF (20)
        # 50-69: SAMP_NUM_COEFF (20)
        # 70-89: SAMP_DEN_COEFF (20)
        
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
        
        self.line_num_coeff = coeffs[..., 10:30]
        self.line_den_coeff = coeffs[..., 30:50]
        self.samp_num_coeff = coeffs[..., 50:70]
        self.samp_den_coeff = coeffs[..., 70:90]

    def _polynomial(self, p, l, h, coeffs):
        # p: latitude (normalized), l: longitude (normalized), h: height (normalized)
        # coeffs: (..., 20)
        # p, l, h can be [..., spatial_dims] while coeffs is [batch, 20]
        # Need to reshape coeffs to match
        
        # Compute spatial_dims to add
        spatial_dims = p.ndim - (coeffs.ndim - 1)  # coeffs.ndim-1 is batch dims
        
        # Extract coefficients and add spatial dimensions
        c = []
        for i in range(20):
            coef = coeffs[..., i]
            if spatial_dims > 0:
                coef = coef.view(*coef.shape, *([1] * spatial_dims))
            c.append(coef)

        result = (c[0] + c[1]*l + c[2]*p + c[3]*h + c[4]*l*p + c[5]*l*h + c[6]*p*h + c[7]*l*l + c[8]*p*p + c[9]*h*h +
                  c[10]*p*l*h + c[11]*l*l*l + c[12]*l*p*p + c[13]*l*h*h + c[14]*l*l*p + c[15]*p*p*p + c[16]*p*h*h + c[17]*l*l*h + c[18]*p*p*h + c[19]*h*h*h)
        return result

    def forward(self, lat: Tensor, lon: Tensor, height: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Project (Lat, Lon, Height) world coordinates to (Row, Col) image coordinates.
        Input are typically tensors of shape (B, ...).
        World coords are un-normalized (deg, deg, meters).
        Returns (row, col) (pixel coordinates).
        """
        # 1. Normalize - handle multi-dimensional inputs
        spatial_dims = lat.ndim - self.lat_off.ndim
        lat_off = self.lat_off.view(*self.lat_off.shape, *([1] * spatial_dims))
        lat_scale = self.lat_scale.view(*self.lat_scale.shape, *([1] * spatial_dims))
        long_off = self.long_off.view(*self.long_off.shape, *([1] * spatial_dims))
        long_scale = self.long_scale.view(*self.long_scale.shape, *([1] * spatial_dims))
        height_off = self.height_off.view(*self.height_off.shape, *([1] * spatial_dims))
        height_scale = self.height_scale.view(*self.height_scale.shape, *([1] * spatial_dims))
        line_off = self.line_off.view(*self.line_off.shape, *([1] * spatial_dims))
        line_scale = self.line_scale.view(*self.line_scale.shape, *([1] * spatial_dims))
        samp_off = self.samp_off.view(*self.samp_off.shape, *([1] * spatial_dims))
        samp_scale = self.samp_scale.view(*self.samp_scale.shape, *([1] * spatial_dims))
        
        P = (lat - lat_off) / lat_scale
        L = (lon - long_off) / long_scale
        H = (height - height_off) / height_scale

        # 2. Compute polynomials
        num_l = self._polynomial(P, L, H, self.line_num_coeff)
        den_l = self._polynomial(P, L, H, self.line_den_coeff)
        num_s = self._polynomial(P, L, H, self.samp_num_coeff)
        den_s = self._polynomial(P, L, H, self.samp_den_coeff)

        # 3. Calculate normalized coordinates
        r_n = num_l / den_l
        c_n = num_s / den_s

        # 4. De-normalize to image coordinates
        row = r_n * line_scale + line_off
        col = c_n * samp_scale + samp_off

        return row, col

    def inverse(self, row: Tensor, col: Tensor, height: Tensor, initial_guess: Tuple[Tensor, Tensor] = None, iterations=5) -> Tuple[Tensor, Tensor]:
        """
        Iterative inverse projection (Newton-Raphson or similar) from (Row, Col, Height) to (Lat, Lon).
        Row, Col are un-normalized pixels. Height is un-normalized meters.
        Returns (lat, lon).
        """
        # Normalize inputs
        # row, col, height can be [B, D, H, W] while self params are [B]
        # Need to reshape params to match: [B, 1, 1, 1]
        spatial_dims = row.ndim - self.line_off.ndim
        line_off = self.line_off.view(*self.line_off.shape, *([1] * spatial_dims))
        line_scale = self.line_scale.view(*self.line_scale.shape, *([1] * spatial_dims))
        samp_off = self.samp_off.view(*self.samp_off.shape, *([1] * spatial_dims))
        samp_scale = self.samp_scale.view(*self.samp_scale.shape, *([1] * spatial_dims))
        height_off = self.height_off.view(*self.height_off.shape, *([1] * spatial_dims))
        height_scale = self.height_scale.view(*self.height_scale.shape, *([1] * spatial_dims))
        lat_off = self.lat_off.view(*self.lat_off.shape, *([1] * spatial_dims))
        lat_scale = self.lat_scale.view(*self.lat_scale.shape, *([1] * spatial_dims))
        long_off = self.long_off.view(*self.long_off.shape, *([1] * spatial_dims))
        long_scale = self.long_scale.view(*self.long_scale.shape, *([1] * spatial_dims))
        
        Rn = (row - line_off) / line_scale
        Cn = (col - samp_off) / samp_scale
        H = (height - height_off) / height_scale

        # Initial guess: use offset (0, 0) normalized, which corresponds to (lat_off, long_off)
        if initial_guess is None:
            # normalized initial guess (0, 0)
            P = torch.zeros_like(Rn)
            L = torch.zeros_like(Cn)
        else:
            lat_guess, lon_guess = initial_guess
            P = (lat_guess - lat_off) / lat_scale
            L = (lon_guess - long_off) / long_scale

        # Newton-Raphson iteration
        # We want to solve for P, L:
        # f1(P, L) = NumL(P,L,H)/DenL(P,L,H) - Rn = 0
        # f2(P, L) = NumS(P,L,H)/DenS(P,L,H) - Cn = 0
        
        for _ in range(iterations):
            # Evaluate polynomials
            # We need gradients w.r.t P and L. 
            # PyTorch autograd can handle this easily if we just compute the forward pass and grab gradients? 
            # But doing autograd inside a no_grad loop or optimizing logic might be slow/complex.
            # For 5 iterations, pure autograd is fine if we are not inside a larger graph we want to backprop through?
            # Actually, we WANT to backprop through this eventually (maybe? usually no, this is just geometry).
            # If we need this to be differentiable for the model (e.g. if we are optimizing coordinates), we should use implicit differentiation or unrolled loop.
            # Here we just implement unrolled loop.
            
            # To use autograd for Jacobian:
            with torch.enable_grad():
                P_var = P.detach().requires_grad_(True)
                L_var = L.detach().requires_grad_(True)
                
                num_l = self._polynomial(P_var, L_var, H, self.line_num_coeff)
                den_l = self._polynomial(P_var, L_var, H, self.line_den_coeff)
                num_s = self._polynomial(P_var, L_var, H, self.samp_num_coeff)
                den_s = self._polynomial(P_var, L_var, H, self.samp_den_coeff)
                
                rn_hat = num_l / den_l
                cn_hat = num_s / den_s
                
                err_r = rn_hat - Rn
                err_c = cn_hat - Cn
                
                # Compute Jacobian
                # J = [[dr/dP, dr/dL], [dc/dP, dc/dL]]
                # This construct is batched.
                
                # dr/dP
                grad_r_p = torch.autograd.grad(rn_hat, P_var, grad_outputs=torch.ones_like(rn_hat), create_graph=True, retain_graph=True)[0]
                grad_r_l = torch.autograd.grad(rn_hat, L_var, grad_outputs=torch.ones_like(rn_hat), create_graph=True, retain_graph=True)[0]
                grad_c_p = torch.autograd.grad(cn_hat, P_var, grad_outputs=torch.ones_like(cn_hat), create_graph=True, retain_graph=True)[0]
                grad_c_l = torch.autograd.grad(cn_hat, L_var, grad_outputs=torch.ones_like(cn_hat), create_graph=True, retain_graph=True)[0]
                
            # Detach gradients to stop graph growth if irrelevant, but keep for next iter if needed
            # For pure solver we don't need graph across steps unless we want gradients of output wrt input.
            # Assuming we might want gradients of LatLon wrt RowCol? 
            # For now, let's stick to simple update.
            
            # Jacobian inverse update:
            # [dP, dL]^T = - J^-1 * [err_r, err_c]^T
            
            det = grad_r_p * grad_c_l - grad_r_l * grad_c_p
            det = det + 1e-10 # eps
            
            inv_j_00 = grad_c_l / det
            inv_j_01 = -grad_r_l / det
            inv_j_10 = -grad_c_p / det
            inv_j_11 = grad_r_p / det
            
            delta_p = -(inv_j_00 * err_r + inv_j_01 * err_c)
            delta_l = -(inv_j_10 * err_r + inv_j_11 * err_c)
            
            P = P_var.detach() + delta_p.detach()
            L = L_var.detach() + delta_l.detach()

        # De-normalize
        lat = P * lat_scale + lat_off
        lon = L * long_scale + long_off

        return lat, lon
