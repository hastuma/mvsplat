import os
import sys
import math
import csv

import torch
import rasterio


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.append(ROOT_DIR)

from src.geometry.rpc import RPC


def build_rpc_from_tiff(tiff_path: str, device: str = "cpu") -> RPC:
	"""從 GeoTIFF 的 RPC metadata 讀取係數並建立 RPC 物件。"""
	with rasterio.open(tiff_path) as src:
		rpc_tags = src.tags(ns="RPC")

	required_keys = [
		"LINE_OFF",
		"SAMP_OFF",
		"LAT_OFF",
		"LONG_OFF",
		"HEIGHT_OFF",
		"LINE_SCALE",
		"SAMP_SCALE",
		"LAT_SCALE",
		"LONG_SCALE",
		"HEIGHT_SCALE",
		"LINE_NUM_COEFF",
		"LINE_DEN_COEFF",
		"SAMP_NUM_COEFF",
		"SAMP_DEN_COEFF",
	]

	for k in required_keys:
		if k not in rpc_tags:
			raise RuntimeError(f"RPC key {k} not found in TIFF metadata")

	# 0-9: offsets & scales
	line_off = float(rpc_tags["LINE_OFF"])
	samp_off = float(rpc_tags["SAMP_OFF"])
	lat_off = float(rpc_tags["LAT_OFF"])
	long_off = float(rpc_tags["LONG_OFF"])
	height_off = float(rpc_tags["HEIGHT_OFF"])

	line_scale = float(rpc_tags["LINE_SCALE"])
	samp_scale = float(rpc_tags["SAMP_SCALE"])
	lat_scale = float(rpc_tags["LAT_SCALE"])
	long_scale = float(rpc_tags["LONG_SCALE"])
	height_scale = float(rpc_tags["HEIGHT_SCALE"])

	def parse_coeffs(key: str):
		vals = rpc_tags[key].replace(",", " ").split()
		if len(vals) != 20:
			raise RuntimeError(f"RPC key {key} expected 20 terms, got {len(vals)}")
		return [float(v) for v in vals]

	line_num = parse_coeffs("LINE_NUM_COEFF")
	line_den = parse_coeffs("LINE_DEN_COEFF")
	samp_num = parse_coeffs("SAMP_NUM_COEFF")
	samp_den = parse_coeffs("SAMP_DEN_COEFF")

	coeffs = torch.zeros(90, dtype=torch.float32)
	coeffs[0] = line_off
	coeffs[1] = line_scale
	coeffs[2] = samp_off
	coeffs[3] = samp_scale
	coeffs[4] = lat_off
	coeffs[5] = lat_scale
	coeffs[6] = long_off
	coeffs[7] = long_scale
	coeffs[8] = height_off
	coeffs[9] = height_scale

	coeffs[10:30] = torch.tensor(line_num, dtype=torch.float32)
	coeffs[30:50] = torch.tensor(line_den, dtype=torch.float32)
	coeffs[50:70] = torch.tensor(samp_num, dtype=torch.float32)
	coeffs[70:90] = torch.tensor(samp_den, dtype=torch.float32)

	coeffs = coeffs.unsqueeze(0)  # [1, 90]

	device_obj = torch.device(device)
	coeffs = coeffs.to(device_obj)
	return RPC(coeffs, device=device_obj)


def compute_height_sensitivity(
	tiff_path: str,
	output_csv_path: str,
	summary_csv_path: str,
	deltas_m=None,
	grid_size: int = 21,
	device: str = "cpu",
):
	"""
	對指定 RPC 影像，分析高度偏移對 inverse warping (lat, lon) 的影響，
	取一個粗網格上的像素，計算 baseline (HEIGHT_OFF) 與各種高度偏移下的經緯度差異，
	並把結果存成 CSV。
	"""
	if deltas_m is None:
		# 0 為 baseline，高度 +2, +5, +10, +15, +20, 以及 -20 公尺
		deltas_m = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, -20.0]

	device_obj = torch.device(device)

	# 建立 RPC model，並取得影像大小
	with rasterio.open(tiff_path) as src:
		height_px, width_px = src.height, src.width

	rpc = build_rpc_from_tiff(tiff_path, device=device)

	# 建立取樣像素網格
	rows_1d = torch.linspace(0, height_px - 1, steps=grid_size, dtype=torch.float32, device=device_obj)
	cols_1d = torch.linspace(0, width_px - 1, steps=grid_size, dtype=torch.float32, device=device_obj)
	rr, cc = torch.meshgrid(rows_1d, cols_1d, indexing="ij")  # [G, G]

	row_flat = rr.reshape(1, -1)  # [1, N]
	col_flat = cc.reshape(1, -1)  # [1, N]

	# baseline 高度 (RPC HEIGHT_OFF)
	h0 = rpc.height_off[0].to(device_obj).float()
	H_base = torch.full_like(row_flat, h0)

	with torch.no_grad():
		lat_base, lon_base = rpc.inverse(row_flat, col_flat, H_base)

	lat_base = lat_base.to(device_obj).float()
	lon_base = lon_base.to(device_obj).float()

	rad = math.pi / 180.0
	r_earth = 6378137.0

	os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

	# 詳細結果：每個像素、每個高度偏移
	with open(output_csv_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"row",
				"col",
				"delta_h_m",
				"dlat_deg",
				"dlon_deg",
				"east_m",
				"north_m",
				"dist_m",
			]
		)

		# 統計用
		summary_rows = []

		# baseline 已經在 lat_base / lon_base 中，從 deltas_m[1:] 開始算差異
		base_lat_flat = lat_base.reshape(-1)
		base_lon_flat = lon_base.reshape(-1)

		for dh in deltas_m[1:]:
			H_cur = torch.full_like(row_flat, h0 + float(dh))
			with torch.no_grad():
				lat_cur, lon_cur = rpc.inverse(row_flat, col_flat, H_cur)

			lat_cur = lat_cur.to(device_obj).float().reshape(-1)
			lon_cur = lon_cur.to(device_obj).float().reshape(-1)

			dlat = lat_cur - base_lat_flat
			dlon = lon_cur - base_lon_flat

			# 轉成東向/北向位移（公尺）
			cos_lat = torch.cos(base_lat_flat * rad)
			east = dlon * rad * r_earth * cos_lat
			north = dlat * rad * r_earth
			dist = torch.sqrt(east * east + north * north)

			# 寫入逐點結果
			for i in range(row_flat.numel()):
				r_val = rr.reshape(-1)[i].item()
				c_val = cc.reshape(-1)[i].item()
				writer.writerow(
					[
						f"{r_val:.3f}",
						f"{c_val:.3f}",
						f"{dh:.3f}",
						f"{dlat[i].item():.9f}",
						f"{dlon[i].item():.9f}",
						f"{east[i].item():.6f}",
						f"{north[i].item():.6f}",
						f"{dist[i].item():.6f}",
					]
				)

			# 統計值
			mean_dist = float(dist.mean().item())
			max_dist = float(dist.max().item())
			median_dist = float(dist.median().item())
			p95_dist = float(torch.quantile(dist, 0.95).item())

			summary_rows.append(
				[
					dh,
					mean_dist,
					max_dist,
					median_dist,
					p95_dist,
				]
			)

	# 儲存 summary
	with open(summary_csv_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"delta_h_m",
				"mean_dist_m",
				"max_dist_m",
				"median_dist_m",
				"p95_dist_m",
			]
		)
		writer.writerows(summary_rows)


def main():
	tiff_path = "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004/JAX_004_006_RGB.tif"

	out_dir = os.path.dirname(os.path.abspath(__file__))
	output_csv = os.path.join(out_dir, "sensitivity_height_JAX_004_006_RGB_detail.csv")
	summary_csv = os.path.join(out_dir, "sensitivity_height_JAX_004_006_RGB_summary.csv")

	# 如果有 GPU 而且你想用，可以把 device 改成 "cuda"
	device = "cpu"

	compute_height_sensitivity(
		tiff_path=tiff_path,
		output_csv_path=output_csv,
		summary_csv_path=summary_csv,
		deltas_m=[0.0, 2.0, 5.0, 10.0, 15.0, 20.0, -20.0],
		grid_size=21,
		device=device,
	)

	print("Saved detail results to:", output_csv)
	print("Saved summary to:", summary_csv)


if __name__ == "__main__":  # 若你用 python tests/sensitivity_height.py 執行，會跑 main()
	main()

