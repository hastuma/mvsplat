#!/usr/bin/env python3
"""
Display and organize mvsplat test results
Shows: metrics, image counts, file locations, and data sizes
"""
import json
import os
from pathlib import Path

def human_readable_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_dir_size(path):
    """Calculate total size of directory"""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif entry.is_dir(follow_symlinks=False):
            total += get_dir_size(entry.path)
    return total

def main():
    test_dir = Path("outputs/test/dfc2019_rpc_training")

    if not test_dir.exists():
        print("❌ Test results not found at:", test_dir)
        return

    print("\n" + "="*60)
    print("  MVSPlat DFC2019 Test Results Summary")
    print("="*60 + "\n")

    # Count images
    images = list(test_dir.glob("**/color/*.png"))
    scene_dirs = [d for d in test_dir.iterdir() if d.is_dir()]

    # Calculate sizes
    total_size = get_dir_size(test_dir)
    images_size = sum(f.stat().st_size for f in images)

    print(f"📊 Test Statistics:")
    print(f"  • Total scenes:     {len(scene_dirs)}")
    print(f"  • Rendered images:  {len(images)}")
    print(f"  • Images size:      {human_readable_size(images_size)}")
    print(f"  • Total size:       {human_readable_size(total_size)}")

    # Load and display metrics
    scores_file = test_dir / "scores_all_avg.json"
    if scores_file.exists():
        with open(scores_file) as f:
            scores = json.load(f)

        print(f"\n📈 Quality Metrics:")
        if "ssim" in scores:
            print(f"  • SSIM:            {scores['ssim']:.6f}")
        if "lpips" in scores:
            print(f"  • LPIPS:           {scores['lpips']:.6f}")
        if "encoder" in scores:
            encoder_data = scores["encoder"]
            if isinstance(encoder_data, list) and len(encoder_data) >= 2:
                print(f"  • Encoder time:    {encoder_data[1]:.4f}s")
        if "decoder" in scores:
            decoder_data = scores["decoder"]
            if isinstance(decoder_data, list) and len(decoder_data) >= 0:
                print(f"  • Decoder time:    {decoder_data[0]:.6f}s")

    # Display file locations
    print(f"\n📁 Output Files:")
    output_files = [
        ("Rendered images",  f"{test_dir}/*/color/*.png"),
        ("SSIM scores",      scores_file),
        ("LPIPS scores",     test_dir / "scores_lpips_all.json"),
        ("Metrics summary",  test_dir / "scores_all_avg.json"),
        ("Timing benchmark",  test_dir / "benchmark.json"),
        ("Peak memory",       test_dir / "peak_memory.json"),
    ]

    for name, path in output_files:
        if isinstance(path, str):
            path = Path(path)
        if "*" in str(path):
            matches = list(test_dir.glob(path.name.lstrip("*/")))
            status = f"✓ {len(matches)} files" if matches else "✗ Not found"
        else:
            status = "✓ Found" if path.exists() else "✗ Not found"
        print(f"  • {name:<20} {status}")

    print(f"\n📂 Base directory: {test_dir}\n")

    # Show individual scene results if available
    ssim_all = test_dir / "scores_ssim_all.json"
    lpips_all = test_dir / "scores_lpips_all.json"

    if ssim_all.exists() or lpips_all.exists():
        print("⭐ Top 5 scenes (by SSIM):")
        if ssim_all.exists():
            with open(ssim_all) as f:
                ssim_scores = json.load(f)
            if isinstance(ssim_scores, list) and len(ssim_scores) > 0:
                top_5 = sorted(enumerate(ssim_scores), key=lambda x: x[1], reverse=True)[:5]
                for idx, (scene_idx, score) in enumerate(top_5, 1):
                    print(f"  {idx}. Scene {scene_idx}: {score:.6f}")

    print("\n" + "="*60)
    print("✅ Test execution completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
