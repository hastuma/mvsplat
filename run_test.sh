#!/bin/bash
# =============================================================================
# Test mvsplat DFC2019 model on validation dataset
# =============================================================================
# Usage:
#   bash run_test.sh [GPU_ID] [CHECKPOINT_PATH] [DATASET_ROOT] [ENABLE_PLY]
#
# Examples:
#   bash run_test.sh 0                                    # Test without PLY
#   bash run_test.sh 6 "" "" true                         # Enable PLY export
#   bash run_test.sh 0 /path/to/checkpoint /path/to/data true
# =============================================================================

GPU_ID="${1:-0}"
CHECKPOINT="${2:-/project/winston/mvsplat/outputs/2026-03-09/00-43-29/checkpoints/epoch_17799-step_17800.ckpt}"
DATASET_ROOT="${3:-/project/winston/datasets/DFC2019/testing}"
ENABLE_PLY="${4:-false}"

echo "============================================="
echo " mvsplat DFC2019 Test"
echo "============================================="
echo " GPU:           ${GPU_ID}"
echo " Checkpoint:    ${CHECKPOINT}"
echo " Dataset:       ${DATASET_ROOT}"
echo " Test set:      ${DATASET_ROOT}/validation/"
echo " Export PLY:    ${ENABLE_PLY}"
echo "============================================="

# Verify files exist
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

if [ ! -d "${DATASET_ROOT}/validation" ]; then
    echo "ERROR: Validation folder not found: ${DATASET_ROOT}/validation/"
    exit 1
fi

source /project/winston/miniconda3/bin/activate mvsplat
cd /project/winston/mvsplat

# Build command with optional PLY export
EXTRA_ARGS=""
if [ "${ENABLE_PLY}" = "true" ] || [ "${ENABLE_PLY}" = "1" ]; then
    EXTRA_ARGS="model.encoder.visualizer.export_ply=true"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.main \
    +experiment=dfc2019 \
    mode=test \
    checkpointing.load="${CHECKPOINT}" \
    "dataset.roots=[${DATASET_ROOT}]" \
    test.compute_scores=true \
    test.save_image=true \
    test.save_video=false \
    test.output_path=outputs/test \
    data_loader.test.batch_size=1 \
    data_loader.test.num_workers=0 \
    ${EXTRA_ARGS}

# Summary
echo ""
echo "============================================="
echo " Test Complete!"
echo "============================================="
TEST_OUTPUT_DIR="outputs/test/dfc2019_rpc_training"
if [ -d "${TEST_OUTPUT_DIR}" ]; then
    IMAGE_COUNT=$(find "${TEST_OUTPUT_DIR}" -name "*.png" 2>/dev/null | wc -l)
    echo " Images saved: ${IMAGE_COUNT}"
    echo " Location: ${TEST_OUTPUT_DIR}"
    if [ -f "${TEST_OUTPUT_DIR}/scores_all_avg.json" ]; then
        echo " Metrics:"
        python3 -c "import json; m=json.load(open('${TEST_OUTPUT_DIR}/scores_all_avg.json')); print(f'   - SSIM:  {m.get(\"ssim\", \"N/A\"):.4f}' if 'ssim' in m else ''); print(f'   - LPIPS: {m.get(\"lpips\", \"N/A\"):.4f}' if 'lpips' in m else '')" 2>/dev/null || echo "   - See scores_all_avg.json for details"
    fi
    if [ -f "${TEST_OUTPUT_DIR}/benchmark.json" ]; then
        echo " Timing saved: benchmark.json"
    fi
fi
echo "============================================="

