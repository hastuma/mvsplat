#!/bin/bash
# 快速访问测试结果和管理
# Usage: bash manage_results.sh [command]

RESULTS_DIR="outputs/test/dfc2019_rpc_training"

show_help() {
    echo "=========================================="
    echo "  MVSPlat 测试結果管理工具"
    echo "=========================================="
    echo ""
    echo "用法: bash manage_results.sh [命令]"
    echo ""
    echo "命令："
    echo "  summary         - 显示测试结果摘要"
    echo "  browse          - 开启文件浏览器查看结果"
    echo "  stats           - 显示详细统计"
    echo "  export-scores   - 导出所有分数为 CSV"
    echo "  cleanup         - 清理测试输出（谨慎使用）"
    echo "  help            - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash manage_results.sh summary"
    echo "  bash manage_results.sh export-scores"
    echo ""
}

show_summary() {
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "❌ 测试结果目录不存在: $RESULTS_DIR"
        return
    fi

    echo ""
    echo "============================================="
    echo "  测试结果摘要"
    echo "============================================="
    python3 show_test_results.py
}

show_stats() {
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "❌ 测试结果目录不存在: $RESULTS_DIR"
        return
    fi

    IMAGE_COUNT=$(find "$RESULTS_DIR" -name "*.png" 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)
    SCENE_COUNT=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

    echo ""
    echo "============================================="
    echo "  详细统计"
    echo "============================================="
    echo "场景总数:        $SCENE_COUNT"
    echo "渲染图像数:      $IMAGE_COUNT"
    echo "总大小:          $TOTAL_SIZE"
    echo ""

    if [ -f "$RESULTS_DIR/scores_all_avg.json" ]; then
        echo "指标详情:"
        python3 -c "
import json
with open('$RESULTS_DIR/scores_all_avg.json') as f:
    data = json.load(f)
    for k, v in data.items():
        if isinstance(v, (int, float)):
            print(f'  {k}: {v}')
        else:
            print(f'  {k}: {v}')
" 2>/dev/null
    fi

    echo ""
    echo "输出文件:"
    ls -lh "$RESULTS_DIR"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
}

export_scores_csv() {
    if [ ! -f "$RESULTS_DIR/scores_ssim_all.json" ]; then
        echo "❌ 分数文件不存在"
        return
    fi

    OUTPUT_FILE="test_results_export_$(date +%Y%m%d_%H%M%S).csv"

    python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
output_file = "$OUTPUT_FILE"

# Load scores
with open(results_dir / "scores_ssim_all.json") as f:
    ssim_scores = json.load(f)

lpips_file = results_dir / "scores_lpips_all.json"
lpips_scores = []
if lpips_file.exists():
    with open(lpips_file) as f:
        lpips_scores = json.load(f)

# Write CSV
with open(output_file, 'w') as f:
    f.write("scene_id,ssim,lpips\n")
    for i, ssim in enumerate(ssim_scores):
        lpips = lpips_scores[i] if i < len(lpips_scores) else "N/A"
        f.write(f"{i},{ssim},{lpips}\n")

print(f"✅ 已导出: {output_file}")
EOF
}

browse_results() {
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "❌ 测试结果目录不存在: $RESULTS_DIR"
        return
    fi

    echo "📁 打开结果目录: $RESULTS_DIR"
    if command -v nautilus &> /dev/null; then
        nautilus "$RESULTS_DIR" &
    elif command -v dolphin &> /dev/null; then
        dolphin "$RESULTS_DIR" &
    elif command -v open &> /dev/null; then
        open "$RESULTS_DIR"
    else
        ls -lh "$RESULTS_DIR"
    fi
}

cleanup_results() {
    read -p "确定要删除所有测试结果吗? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$RESULTS_DIR"
        echo "✅ 已清理"
    else
        echo "❌ 取消"
    fi
}

# Main
case "${1:-help}" in
    summary)
        show_summary
        ;;
    stats)
        show_stats
        ;;
    export-scores)
        export_scores_csv
        ;;
    browse)
        browse_results
        ;;
    cleanup)
        cleanup_results
        ;;
    help|*)
        show_help
        ;;
esac
