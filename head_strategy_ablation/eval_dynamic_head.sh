#!/bin/bash

# Dynamic Head Selection Evaluation Script
# 测试层次化动态选择策略的效果

# 基础配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 模型和数据路径
MODEL_BASE="lmsys/vicuna-7b-v1.5"
MODEL_PATH="/path/to/your/llava-v1.5-7b"
DATA_PATH="/path/to/evaluation/data"

# 动态选择配置
VISUAL_TOKEN_NUMS=(192 128 64)
MIN_HEADS=6
MAX_HEADS=24
ENABLE_DEBUG=true

# 评估基准
BENCHMARKS=("textvqa" "mme" "gqa" "pope")

# GPU配置
CHUNKS=8
GPULIST=$(seq 0 $((CHUNKS-1)) | tr '\n' ',' | sed 's/,$//')

echo "🚀 Starting Dynamic Head Selection Evaluation"
echo "📊 Configuration:"
echo "   - Visual tokens: ${VISUAL_TOKEN_NUMS[*]}"
echo "   - Head range: [$MIN_HEADS, $MAX_HEADS]"
echo "   - Debug mode: $ENABLE_DEBUG"
echo "   - GPUs: $GPULIST (${CHUNKS} chunks)"
echo "   - Benchmarks: ${BENCHMARKS[*]}"

# 创建结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="dynamic_head_results_${TIMESTAMP}"
mkdir -p $RESULT_DIR

# 为每个基准和token数量组合运行测试
for BENCHMARK in "${BENCHMARKS[@]}"; do
    for TOKEN_NUM in "${VISUAL_TOKEN_NUMS[@]}"; do

        echo ""
        echo "🎯 Testing: $BENCHMARK with $TOKEN_NUM tokens"
        echo "=================================================="

        # 生成结果文件名
        RESULT_FILE="${RESULT_DIR}/dynamic_${BENCHMARK}_${TOKEN_NUM}_results.txt"

        # 根据基准选择评估脚本和参数
        case $BENCHMARK in
            "textvqa")
                EVAL_SCRIPT="llava.eval.model_vqa_loader"
                EVAL_ARGS="--model-path $MODEL_PATH \\
                          --model-base $MODEL_BASE \\
                          --question-file $DATA_PATH/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \\
                          --image-folder $DATA_PATH/eval/textvqa/train_images \\
                          --answers-file ${RESULT_DIR}/textvqa_${TOKEN_NUM}_answers.jsonl"
                POST_EVAL="python llava/eval/eval_textvqa.py \\
                          --annotation-file $DATA_PATH/eval/textvqa/TextVQA_0.5.1_val.json \\
                          --result-file ${RESULT_DIR}/textvqa_${TOKEN_NUM}_answers.jsonl"
                ;;

            "mme")
                EVAL_SCRIPT="llava.eval.model_vqa_mme"
                EVAL_ARGS="--model-path $MODEL_PATH \\
                          --model-base $MODEL_BASE \\
                          --question-file $DATA_PATH/eval/MME/llava_mme.jsonl \\
                          --image-folder $DATA_PATH/eval/MME/MME_Benchmark_release_version \\
                          --answers-file ${RESULT_DIR}/mme_${TOKEN_NUM}_answers.jsonl \\
                          --single-pred-prompt"
                POST_EVAL="cd $DATA_PATH/eval/MME && python calculation.py --results_dir ${PWD}/${RESULT_DIR}/"
                ;;

            "gqa")
                EVAL_SCRIPT="llava.eval.model_vqa_loader"
                EVAL_ARGS="--model-path $MODEL_PATH \\
                          --model-base $MODEL_BASE \\
                          --question-file $DATA_PATH/eval/gqa/llava_gqa_testdev_balanced.jsonl \\
                          --image-folder $DATA_PATH/eval/gqa/images \\
                          --answers-file ${RESULT_DIR}/gqa_${TOKEN_NUM}_answers.jsonl"
                POST_EVAL="python llava/eval/eval_gqa.py \\
                          --annotation-file $DATA_PATH/eval/gqa/testdev_balanced_questions.json \\
                          --result-file ${RESULT_DIR}/gqa_${TOKEN_NUM}_answers.jsonl"
                ;;

            "pope")
                EVAL_SCRIPT="llava.eval.model_vqa_loader"
                EVAL_ARGS="--model-path $MODEL_PATH \\
                          --model-base $MODEL_BASE \\
                          --question-file $DATA_PATH/eval/pope/llava_pope_test.jsonl \\
                          --image-folder $DATA_PATH/eval/pope/val2014 \\
                          --answers-file ${RESULT_DIR}/pope_${TOKEN_NUM}_answers.jsonl"
                POST_EVAL="python llava/eval/eval_pope.py \\
                          --annotation-dir $DATA_PATH/eval/pope/coco \\
                          --question-file $DATA_PATH/eval/pope/llava_pope_test.jsonl \\
                          --result-file ${RESULT_DIR}/pope_${TOKEN_NUM}_answers.jsonl"
                ;;
        esac

        echo "📝 Logging to: $RESULT_FILE"

        # 运行评估
        {
            echo "=========================================="
            echo "DYNAMIC HEAD SELECTION EVALUATION"
            echo "Benchmark: $BENCHMARK"
            echo "Visual Tokens: $TOKEN_NUM"
            echo "Timestamp: $(date)"
            echo "Min Heads: $MIN_HEADS"
            echo "Max Heads: $MAX_HEADS"
            echo "Debug Mode: $ENABLE_DEBUG"
            echo "=========================================="
            echo ""

            # 设置动态选择参数
            export VISUAL_TOKEN_NUM=$TOKEN_NUM
            export MIN_HEADS=$MIN_HEADS
            export MAX_HEADS=$MAX_HEADS
            export ENABLE_DEBUG=$ENABLE_DEBUG

            # 运行模型评估
            python -m $EVAL_SCRIPT \\
                $EVAL_ARGS \\
                --num-chunks $CHUNKS \\
                --chunk-idx 0 \\
                --temperature 0 \\
                --conv-mode vicuna_v1 \\
                --visual-token-num $TOKEN_NUM \\
                --enable-dynamic-selection true \\
                --min-heads $MIN_HEADS \\
                --max-heads $MAX_HEADS \\
                --debug-dynamic-selection $ENABLE_DEBUG

            echo ""
            echo "🔍 Running post-evaluation..."

            # 运行后处理评估
            eval $POST_EVAL

            echo ""
            echo "✅ Completed: $BENCHMARK with $TOKEN_NUM tokens"
            echo "=========================================="

        } 2>&1 | tee -a $RESULT_FILE

        echo "📊 Results saved to: $RESULT_FILE"

    done
done

echo ""
echo "🎉 All evaluations completed!"
echo "📁 Results directory: $RESULT_DIR"
echo ""
echo "📋 Summary of generated files:"
ls -la $RESULT_DIR/

# 生成汇总报告
echo ""
echo "📝 Generating summary report..."

SUMMARY_FILE="${RESULT_DIR}/dynamic_head_summary.md"

cat > $SUMMARY_FILE << EOF
# Dynamic Head Selection Evaluation Summary

**Evaluation Date**: $(date)
**Configuration**:
- Visual Token Numbers: ${VISUAL_TOKEN_NUMS[*]}
- Head Range: [$MIN_HEADS, $MAX_HEADS]
- Debug Mode: $ENABLE_DEBUG
- GPU Configuration: $CHUNKS chunks

## Results Overview

EOF

# 解析每个基准的结果
for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "### $BENCHMARK Results" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    echo "| Visual Tokens | Score | Selected Heads (avg) | Primary Strategy |" >> $SUMMARY_FILE
    echo "|---------------|-------|---------------------|------------------|" >> $SUMMARY_FILE

    for TOKEN_NUM in "${VISUAL_TOKEN_NUMS[@]}"; do
        RESULT_FILE="${RESULT_DIR}/dynamic_${BENCHMARK}_${TOKEN_NUM}_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            # 这里可以添加结果解析逻辑
            echo "| $TOKEN_NUM | TBD | TBD | TBD |" >> $SUMMARY_FILE
        fi
    done

    echo "" >> $SUMMARY_FILE
done

echo "✅ Summary report generated: $SUMMARY_FILE"

# 创建快速测试脚本
cat > "${RESULT_DIR}/quick_test.sh" << 'EOF'
#!/bin/bash
# Quick test script for dynamic head selection

echo "🔧 Quick Dynamic Head Selection Test"

# 单个样本测试
python -c "
import torch
from llava.model.language_model.modeling_llama_fastv_ablation_dynamic import FineFastVLlamaModelAblationDynamic

# 模拟注意力数据
image_attention = torch.randn(32, 576)  # 32 heads, 576 visual tokens

# 创建配置
class MockConfig:
    def __init__(self):
        self.num_hidden_layers = 32
        self.image_aspect_ratio = 'pad'
        self.enable_dynamic_selection = True
        self.min_heads = 6
        self.max_heads = 24
        self.debug_dynamic_selection = True
        self.H = 16

config = MockConfig()

# 测试动态选择
model = FineFastVLlamaModelAblationDynamic.__new__(FineFastVLlamaModelAblationDynamic)
model.config = config
model.min_heads = 6
model.max_heads = 24
model.H = 16
model.debug_mode = True

# 运行动态选择
selected_heads, strategy = model.hierarchical_dynamic_selection(image_attention)

print(f'✅ Test completed!')
print(f'   Selected {len(selected_heads)} heads using {strategy} strategy')
print(f'   Head indices: {selected_heads.tolist()}')
"
EOF

chmod +x "${RESULT_DIR}/quick_test.sh"

echo ""
echo "🚀 Quick test script created: ${RESULT_DIR}/quick_test.sh"
echo "   Run it to test the dynamic selection logic independently"

echo ""
echo "🎯 Next Steps:"
echo "1. Review results in: $RESULT_DIR"
echo "2. Compare with baseline static head selection"
echo "3. Analyze the selected strategies for different scenarios"
echo "4. Run quick test: cd $RESULT_DIR && ./quick_test.sh"