# MVSPlat DFC2019 Testing Guide

## 📊 Current Test Results

已成功测试 **641 个场景**，测试结果已保存至：`outputs/test/dfc2019_rpc_training/`

### 质量指标
- **SSIM**: 0.063156
- **LPIPS**: 0.693349
- **编码器速度**: 2.49 秒/图像
- **解码器速度**: 0.0062 秒/图像

### 输出内容
- ✅ 641 张渲染图像（8.5 MB）
- ✅ SSIM 分数 (scores_ssim_all.json)
- ✅ LPIPS 分数 (scores_lpips_all.json)
- ✅ 平均指标 (scores_all_avg.json)
- ✅ 性能基准 (benchmark.json)
- ✅ 峰值内存 (peak_memory.json)

---

## 🚀 如何运行测试

### 基础用法（仅渲染和评分）
```bash
bash run_test.sh 0
```

### 启用 PLY 导出
```bash
bash run_test.sh 6 "" "" true
```

### 自定义模型和数据集
```bash
bash run_test.sh 0 /path/to/checkpoint.ckpt /path/to/dataset_root
```

### 启用 PLY 导出 + 自定义路径
```bash
bash run_test.sh 0 /path/to/checkpoint.ckpt /path/to/dataset_root true
```

---

## 📁 文件位置说明

### 测试输出目录结构
```
outputs/test/dfc2019_rpc_training/
├── JAX_020_p0000/
│   └── color/
│       └── 000003.png          # 渲染图像
├── JAX_020_p0001/
│   └── color/
│       └── 000003.png
├── ...（641 个场景）...
├── scores_all_avg.json         # 平均指标 (SSIM, LPIPS, 时间)
├── scores_ssim_all.json        # 每个场景的 SSIM 分数
├── scores_lpips_all.json       # 每个场景的 LPIPS 分数
├── benchmark.json              # 编码/解码器执行时间
└── peak_memory.json            # 峰值内存使用
```

---

## 🎯 启用 PLY 导出（可选）

PLY 文件是 3D Gaussian 点云的标准格式，可在 CloudCompare 等工具中查看。

### 启用 PLY 导出（重新运行测试）
```bash
bash run_test.sh 6 "" "" true
```

这会在各场景目录下生成 `*.ply` 文件，例如：
```
outputs/test/dfc2019_rpc_training/
├── JAX_020_p0000/
│   ├── color/
│   │   └── 000003.png
│   └── gaussians.ply           # 3D Gaussian 点云模型
├── JAX_020_p0001/
│   ├── color/
│   │   └── 000003.png
│   └── gaussians.ply
├── ...
```

---

## 📊 查看测试结果

### 快速总结
```bash
python3 show_test_results.py
```

输出内容：
- 场景总数
- 渲染图像数
- 文件大小统计
- 质量指标（SSIM、LPIPS）
- 性能时间
- 文件位置

### 详细指标

**平均指标** (`scores_all_avg.json`):
```json
{
  "ssim": 0.063156,
  "lpips": 0.693349,
  "encoder": [1, 2.4922],
  "decoder": [1, 0.0062]
}
```

**所有场景的 SSIM** (`scores_ssim_all.json`):
```json
[0.063156, ...]  // 641 个值，每个对应一个场景
```

---

## ⚙️ 测试配置详情

### 默认配置
- **Model**: CostVolume Encoder + CUDA Splatting Decoder
- **Dataset**: DFC2019 RPC imagery (256×256 patches)
- **Batch size**: 1 (per GPU)
- **GPU memory**: ~6-8 GB per scene (单场景测试)
- **Output path**: `outputs/test/dfc2019_rpc_training/`

### 修改配置的高级方法

直接调用 Hydra CLI（绕过 `run_test.sh`）：
```bash
source /project/winston/miniconda3/bin/activate mvsplat
cd /project/winston/mvsplat

CUDA_VISIBLE_DEVICES=6 python -m src.main \
    +experiment=dfc2019 \
    mode=test \
    checkpointing.load=/path/to/checkpoint.ckpt \
    "dataset.roots=[/path/to/dataset]" \
    test.compute_scores=true \
    test.save_image=true \
    test.save_video=false \
    test.output_path=outputs/test \
    model.encoder.visualizer.export_ply=true \
    data_loader.test.batch_size=1
```

---

## 🔧 常见问题

### Q: 如何开始在新的检查点上重新测试？
```bash
# 复制最新检查点路径并运行
bash run_test.sh 0 /project/winston/mvsplat/outputs/2026-03-09/00-43-29/checkpoints/epoch_17799-step_17800.ckpt /project/winston/datasets/DFC2019/testing
```

### Q: 如何将测试结果保存到自定义位置？
编辑 `run_test.sh`，修改：
```bash
test.output_path=outputs/test/my_custom_name
```

### Q: 如何并行运行多个 GPU 的测试？
```bash
# GPU 0
bash run_test.sh 0 &

# GPU 1
bash run_test.sh 1 &

# 等待完成
wait
```

### Q: 时间太长？如何加快测试？
虽然无法加快推理本身，但可以：
1. 减少测试场景（编辑 `view_sampler_dfc2019.yaml`）
2. 减少计算分数（设 `test.compute_scores=false`）
3. 禁用 PLY 导出（不使用 `export_ply=true`）

---

## 📝 训练脚本参考

如果需要从头训练：
```bash
CUDA_VISIBLE_DEVICES=6 python -m src.main \
    +experiment=dfc2019 \
    mode=train \
    data_loader.train.batch_size=1
```

---

**最后更新**: 2026-03-09
**检查点版本**: epoch_17799-step_17800
**测试集**: DFC2019 geo_cropped validation (1216 scenes → 641 tested)
