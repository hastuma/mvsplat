# /project/winston/mvsplat/outputs/ctx0.png
# /project/winston/mvsplat/outputs/ctx1.png
# /project/winston/mvsplat/outputs/ctx2.png


    # "col_start": 256,
    # "row_start": 512

 
    # "col_start": 339,
    # "row_start": 526


    #   "col_start": 398,
    # "row_start": 547
from PIL import Image, ImageDraw

# 圖片路徑
image_paths = {
    # "ctx0": "/project/winston/mvsplat/outputs/ctx0.png",
    # "ctx1": "/project/winston/mvsplat/outputs/ctx1.png",
    # "ctx2": "/project/winston/mvsplat/outputs/ctx2.png",
    "ctx0": "/project/winston/datasets/DFC2019/Track3-RGB-1/JAX_004_007_RGB.tif",
    "ctx1": "/project/winston/datasets/DFC2019/Track3-RGB-1/JAX_004_014_RGB.tif",
    "ctx2": "/project/winston/datasets/DFC2019/Track3-RGB-1/JAX_004_015_RGB.tif",
    
}

# 對應座標（左上角）
coordinates = {
    "ctx0": {"col_start": 256, "row_start": 512},
    "ctx1": {"col_start": 339, "row_start": 526},
    "ctx2": {"col_start": 398, "row_start": 547},
}

# 方框大小
BOX_SIZE = 256

for key in image_paths:
    img_path = image_paths[key]
    coord = coordinates[key]

    # 開啟圖片
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 左上角
    x1 = coord["col_start"]
    y1 = coord["row_start"]

    # 右下角
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    # 畫紅色框框（width 可以調整粗細）
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # 存檔（避免覆蓋原圖）
    output_path = img_path.replace(".png", "_boxed.png")
    image.save(output_path)

    print(f"Saved: {output_path}")