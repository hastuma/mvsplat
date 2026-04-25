import os
from PIL import Image, ImageDraw, ImageFont

# === 設定 ===
folder = "mvsplat/debug/cropped"
output_path = "mvsplat/debug/cropped/concat.png"

img_size = 256
grid_size = 3
label_height = 30  # 上方標籤高度
padding = 10       # 每格間距

# === 讀取圖片 ===
image_files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

images = []
for i, fname in enumerate(image_files[:7]):  # 只取前7張
    img = Image.open(os.path.join(folder, fname)).convert("RGB")
    img = img.resize((img_size, img_size))
    images.append((img, f"view_{i}"))

# 補空白到9張
while len(images) < 9:
    blank = Image.new("RGB", (img_size, img_size), (220, 220, 220))
    images.append((blank, ""))

# === 計算整張大小 ===
cell_w = img_size
cell_h = img_size + label_height

canvas_w = grid_size * cell_w + (grid_size + 1) * padding
canvas_h = grid_size * cell_h + (grid_size + 1) * padding

canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# 字體（如果沒有會 fallback）
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

# === 貼圖 ===
for idx, (img, label) in enumerate(images):
    row = idx // grid_size
    col = idx % grid_size

    x = padding + col * (cell_w + padding)
    y = padding + row * (cell_h + padding)

    # 畫 label（在圖片上方）
    if label:
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        text_x = x + (cell_w - text_w) // 2
        text_y = y + (label_height - text_h) // 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    # 貼圖片（在 label 下方）
    canvas.paste(img, (x, y + label_height))

# === 存檔 ===
canvas.save(output_path)
print(f"Saved to {output_path}")