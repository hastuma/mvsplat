import cv2
import os

IMAGE_DIR = "/project/winston/mvsplat/outputs/2026-04-24/01-58-22/vis_damv2"
OUTPUT_PATH = "/project/winston/mvsplat/outputs/2026-04-24/01-58-22/depth.mp4"
FPS = 130

frames = []
for step in range(100, 19900, 10):
    path = os.path.join(IMAGE_DIR, f"step_{step:06d}_ctx0.png")
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: missing {path}")
        continue
    frames.append(img)

if not frames:
    raise RuntimeError("No frames found")

fh, fw = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))
for frame in frames:
    writer.write(frame)
writer.release()
print(f"Saved {len(frames)} frames to {OUTPUT_PATH}")
