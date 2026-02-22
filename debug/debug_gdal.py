from osgeo import gdal
import os
import sys

def verify_rpc(input_path, expected_line_off_max=200):
    print(f"正在讀取影像: {input_path}")
    src_ds = gdal.Open(str(input_path))
    if src_ds is None:
        print("錯誤: 無法開啟影像檔案。")
        return False

    # 1. 取得 RPC 參數
    rpc_md = src_ds.GetMetadata('RPC')
    print("\n--- RPC 檢查 ---")
    line_off = float(rpc_md.get('LINE_OFF', '0'))
    samp_off = float(rpc_md.get('SAMP_OFF', '0'))
    
    print(f"LINE_OFF: {line_off}")
    print(f"SAMP_OFF: {samp_off}")

    # 128 是 256/2。如果 correction 成功，offset 應該接近 patch 中心。
    # 之前是 ~19000。
    is_valid = True
    if line_off > expected_line_off_max:
        print(f"警告：LINE_OFF ({line_off}) 過大！可能仍為 Global Coordinates。")
        is_valid = False
    
    if samp_off > expected_line_off_max:
         print(f"警告：SAMP_OFF ({samp_off}) 過大！可能仍為 Global Coordinates。")
         is_valid = False

    if is_valid:
        print(">>> 成功：RPC 參數看起來已是 Local Coordinates。")
    return is_valid

if __name__ == "__main__":
    # Check one of the regenerated files
    INPUT_IMG = "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0002/JAX_004_009_RGB.tif"
    verify_rpc(INPUT_IMG, expected_line_off_max=5000) # give some leeway