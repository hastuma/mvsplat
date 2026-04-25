import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

JSON_PATH = Path(__file__).parent / "p0201_crop_offsets.json"
IMG_DIR = Path(__file__).parent.parent.parent / "datasets/DFC2019/Track3-RGB-1"
OUT_DIR = Path(__file__).parent / "marked"
CROP_DIR = Path(__file__).parent / "cropped"
BOX_SIZE = 256
LINE_WIDTH = 3

OUT_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

with open(JSON_PATH) as f:
    offsets = json.load(f)

for name, info in offsets.items():
    img_path = IMG_DIR / f"{name}.tif"
    if not img_path.exists():
        print(f"[SKIP] {img_path} not found")
        continue

    img = Image.open(img_path).convert("RGB")

    x0, y0 = info["col_start"], info["row_start"]
    x1, y1 = x0 + BOX_SIZE, y0 + BOX_SIZE

    # crop the specified region
    cropped = img.crop((x0, y0, x1, y1))
    crop_path = CROP_DIR / f"{name}_crop.png"
    cropped.save(crop_path)
    print(f"[CROP] {crop_path}")

    # draw red bounding box on original
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=LINE_WIDTH)

    out_path = OUT_DIR / f"{name}_marked.png"
    img.save(out_path)
    print(f"[OK]   {out_path}")
