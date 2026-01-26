
import sys
import torch
import numpy as np
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from geometry.rpc import RPC

def test_rpc_mock_roundtrip():
    # Coefficients from JAX_004_006_RGB.tif (Step 19)
    # LINE_OFF=21376.7839876204, LINE_SCALE=22758
    # SAMP_OFF=20480.9463351176, SAMP_SCALE=21250
    # LAT_OFF=30.3003, LAT_SCALE=0.0703
    # LONG_OFF=-81.6408, LONG_SCALE=0.0729
    # HEIGHT_OFF=-21, HEIGHT_SCALE=501
    
    # We construct the tensor manually.
    # Order: [LINE_OFF, LINE_SCALE, SAMP_OFF, SAMP_SCALE, LAT_OFF, LAT_SCALE, LONG_OFF, LONG_SCALE, HEIGHT_OFF, HEIGHT_SCALE]
    # Followed by 20 LINE_NUM, 20 LINE_DEN, 20 SAMP_NUM, 20 SAMP_DEN
    
    coeffs = torch.zeros(90, dtype=torch.float32)
    
    coeffs[0] = 21376.7839876204
    coeffs[1] = 22758
    coeffs[2] = 20480.9463351176
    coeffs[3] = 21250
    coeffs[4] = 30.3003
    coeffs[5] = 0.0703
    coeffs[6] = -81.6408
    coeffs[7] = 0.0729
    coeffs[8] = -21
    coeffs[9] = 501
    
    # For coefficients vectors, I'll copy a few significant ones and set others to 0 or IDENTITY-like?
    # Actually, "1 0.00... " indicates the first term (constant) is 1.
    # LINE_DEN_COEFF=1 ...
    # SAMP_DEN_COEFF=1 ...
    # LINE_NUM_COEFF=-9.59e-5 ... (First term is constant? No, usually first term is 1 for denominators, but numerators can be anything)
    
    # Wait, copying 80 coefficients by hand is error prone.
    # But I valid RPCs are needed for valid round trip.
    # I will try to parse the string from Step 19.
    
    line_den_str = "1 0.0001684873 -6.335265e-05 1.507818e-05 4.833381e-06 1.910986e-07 1.795369e-06 1.114958e-05 -3.57498e-05 1.553184e-05 3.877905e-07 0 -1.523022e-05 0 4.164689e-07 0.0001748632 8.989371e-08 0 -6.761896e-06 0"
    line_num_str = "-9.59544e-05 0.03701784 -1.051146 0.01355396 0.0001434057 -2.877661e-06 1.275568e-05 -0.0004139917 8.666037e-05 -1.204151e-06 -3.238694e-07 5.050705e-07 3.533154e-06 5.778745e-07 -1.224662e-05 -5.844016e-05 -1.634623e-05 1.783486e-07 1.085339e-06 2.096943e-07"
    samp_den_str = "1 0.001049638 0.0009599374 -0.0004588982 -1.043929e-05 -8.670581e-07 1.354862e-07 5.755182e-06 3.026614e-07 -2.442154e-06 -2.770866e-08 0 1.838139e-07 0 -1.399761e-07 1.059721e-07 0 0 0 0"
    samp_num_str = "-0.004379428 1.016398 0.0008397763 -0.01385913 -0.0009450556 0.000391861 -0.0002151448 0.00335282 -0.000251548 -1.929016e-06 -1.477436e-06 4.961405e-06 -4.221051e-06 -2.19379e-06 1.754333e-05 -3.721629e-05 -9.691363e-08 2.320462e-06 1.804481e-06 3.192365e-08"
    
    def parsestr(s, offset, c):
        vals = [float(x) for x in s.split()]
        for i, v in enumerate(vals):
            c[offset+i] = v
            
    parsestr(line_num_str, 10, coeffs)
    parsestr(line_den_str, 30, coeffs)
    parsestr(samp_num_str, 50, coeffs)
    parsestr(samp_den_str, 70, coeffs)
    
    rpc_obj = RPC(coeffs.unsqueeze(0))
    
    # Test Center
    print("Testing RPC mock roundtrip...")
    H_val = 0.0 # meters (relative to offset? No, input is absolute meters, RPC class normalizes it)
    # Wait, RPC class input height: "H = (height - self.height_off) / self.height_scale"
    # So we pass absolute height.
    
    lat, lon = rpc_obj.inverse(torch.tensor([1024.0]), torch.tensor([1024.0]), torch.tensor([H_val]), iterations=10)
    print(f"I(1024, 1024, {H_val}) -> Lat: {lat.item()}, Lon: {lon.item()}")
    
    r, c = rpc_obj.forward(lat, lon, torch.tensor([H_val]))
    print(f"F(Lat, Lon, {H_val}) -> Row: {r.item()}, Col: {c.item()}")
    
    err_r = abs(r.item() - 1024.0)
    err_c = abs(c.item() - 1024.0)
    print(f"Errors: {err_r}, {err_c}")
    
    assert err_r < 1.0, f"Row error {err_r} too big"
    assert err_c < 1.0, f"Col error {err_c} too big"

if __name__ == "__main__":
    test_rpc_mock_roundtrip()
