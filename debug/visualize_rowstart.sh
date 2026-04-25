#!/bin/bash

# 啟動 conda 環境
source /project/winston/miniconda3/bin/activate mvsplat

# 設定變數
CROP_OFFSET_FILE="/project/winston/datasets/DFC2019/overfit/training/JAX_004_007_p0201/crop_offsets.json"
OUTPUT_BASE="/project/winston/mvsplat/outputs/2026-04-05"
VISUALIZE_DIR="/project/winston/mvsplat/outputs/visualize_row"
INITIAL_ROW_START=547
INCREMENT=2
NUM_ITERATIONS=50
MVSPLAT_DIR="/project/winston/mvsplat"

# 建立 visualize 資料夾（如果不存在）
mkdir -p "$VISUALIZE_DIR"

# 儲存所有生成的資料夾路徑
declare -a GENERATED_DIRS

echo "開始執行循環，總共 $NUM_ITERATIONS 次..."

# 循環 50 次
for i in $(seq 1 $NUM_ITERATIONS); do
    # 計算新的 row_start 值
    NEW_ROW_START=$((INITIAL_ROW_START + (i - 1) * INCREMENT))
    
    echo "=========================================="
    echo "迭代 $i / $NUM_ITERATIONS"
    echo "設定 row_start = $NEW_ROW_START"
    echo "=========================================="
    
    # 修改 crop_offsets.json 中的 row_start 值
    # 使用 jq 修改 JSON 文件
    jq ".\"JAX_004_015_RGB\".row_start = $NEW_ROW_START" "$CROP_OFFSET_FILE" > "${CROP_OFFSET_FILE}.tmp"
    mv "${CROP_OFFSET_FILE}.tmp" "$CROP_OFFSET_FILE"
    
    echo "已更新 crop_offsets.json，row_start = $NEW_ROW_START"
    
    # 執行訓練指令
    echo "執行訓練指令..."
    cd "$MVSPLAT_DIR"
    CUDA_VISIBLE_DEVICES=4 python -m src.main +experiment=dfc2019 data_loader.train.batch_size=1
    
    # 取得最新生成的時間戳資料夾
    LATEST_DIR=$(ls -td "$OUTPUT_BASE"/*/ 2>/dev/null | head -1)
    
    if [ -d "$LATEST_DIR" ]; then
        echo "找到輸出資料夾: $LATEST_DIR"
        GENERATED_DIRS+=("$LATEST_DIR")
        
        # 複製 step_000000_ctx2.png
        SOURCE_IMAGE="$LATEST_DIR/vis_context/step_000000_ctx2.png"
        if [ -f "$SOURCE_IMAGE" ]; then
            DEST_IMAGE="$VISUALIZE_DIR/step_000000_ctx2_iter${i}_row_start_${NEW_ROW_START}.png"
            cp "$SOURCE_IMAGE" "$DEST_IMAGE"
            echo "已複製圖片到: $DEST_IMAGE"
        else
            echo "警告: 找不到 $SOURCE_IMAGE"
        fi
    else
        echo "警告: 找不到輸出資料夾在 $OUTPUT_BASE"
    fi
    
    echo ""
done

echo "=========================================="
echo "所有 50 次迭代完成！"
echo "=========================================="
echo "提取的圖片已存放在: $VISUALIZE_DIR"
echo "正在刪除臨時資料夾..."

# 刪除所有生成的資料夾
for dir in "${GENERATED_DIRS[@]}"; do
    echo "刪除: $dir"
    rm -rf "$dir"
done

echo "清理完成！"

# 恢復原始的 row_start 值
echo "正在恢復原始的 crop_offsets.json..."
jq ".\"JAX_004_015_RGB\".row_start = $INITIAL_ROW_START" "$CROP_OFFSET_FILE" > "${CROP_OFFSET_FILE}.tmp"
mv "${CROP_OFFSET_FILE}.tmp" "$CROP_OFFSET_FILE"
echo "✅ crop_offsets.json 已恢復，row_start = $INITIAL_ROW_START"

# ========== 製作影片 ==========
echo ""
echo "=========================================="
echo "開始製作影片..."
echo "=========================================="

python /project/winston/mvsplat/debug/make_video.py "$VISUALIZE_DIR" "/project/winston/mvsplat/outputs/visualize_row.mp4" 10

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
