#!/bin/bash

# ========== 动态Head选择评估脚本 ==========
echo "🧪 开始动态Head选择评估"

# ========== 基础配置 ==========
WORK_DIR="/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/code/VTP/LLaVA-FinePruner-Head"
CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="dynamic_head"

# 动态选择参数
TOKEN_NUMS=(192 128 64)
MIN_HEADS=6
MAX_HEADS=24

# GPU配置
CHUNKS=8
GPULIST="0,1,2,3,4,5,6,7"

# ========== 帮助信息 ==========
show_usage() {
    echo "用法: $0 [benchmark] [options]"
    echo ""
    echo "基准测试:"
    echo "  pope     - POPE评估"
    echo "  mme      - MME评估"
    echo "  textvqa  - TextVQA评估"
    echo "  gqa      - GQA评估"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  --tokens N     指定visual token数量 (默认: 192,128,64)"
    echo "  --min-heads N  最少头数量 (默认: 6)"
    echo "  --max-heads N  最多头数量 (默认: 24)"
    echo ""
    echo "示例:"
    echo "  $0 pope                    # 运行POPE评估"
    echo "  $0 textvqa --tokens 192    # 只用192个token运行TextVQA"
    echo "  $0 mme --min-heads 8       # 最少使用8个head运行MME"
}

# ========== 参数解析 ==========
BENCHMARK=""
CUSTOM_TOKENS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        pope|mme|textvqa|gqa)
            BENCHMARK="$1"
            shift
            ;;
        --tokens)
            CUSTOM_TOKENS="$2"
            shift 2
            ;;
        --min-heads)
            MIN_HEADS="$2"
            shift 2
            ;;
        --max-heads)
            MAX_HEADS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$BENCHMARK" ]]; then
    echo "❌ 错误: 必须指定benchmark"
    echo ""
    show_usage
    exit 1
fi

# 设置token数量
if [[ -n "$CUSTOM_TOKENS" ]]; then
    TOKEN_NUMS=($CUSTOM_TOKENS)
fi

# ========== 环境初始化 ==========
setup_environment() {
    echo "切换到工作目录: ${WORK_DIR}"
    cd "${WORK_DIR}" || { echo "❌ 错误: 无法切换到工作目录"; exit 1; }

    # 验证关键目录
    echo "验证关键目录..."
    for dir in "playground/data/eval" "llava/eval"; do
        [ ! -d "${dir}" ] && { echo "❌ 错误: 关键目录不存在: ${dir}"; exit 1; }
    done

    echo "✅ 目录结构验证完成"
}

# ========== GPU设置 ==========
setup_gpu() {
    export CUDA_VISIBLE_DEVICES=${GPULIST}
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    echo "🔧 GPU配置: ${GPULIST} (${CHUNKS} chunks)"
}

# ========== 评估任务 ==========
# POPE评估
eval_pope() {
    local TOKEN_NUM=$1
    local RESULT_FILE="${WORK_DIR}/head_strategy_pope_dynamic_${TOKEN_NUM}_results.txt"

    echo "🎯 执行POPE评估 (Token: ${TOKEN_NUM})"

    {
        echo "TOKEN=${TOKEN_NUM}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:"
        python -m llava.eval.model_vqa_loader \\
            --model-path ${CKPT_DIR}/${CKPT} \\
            --model-base lmsys/vicuna-7b-v1.5 \\
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \\
            --image-folder ./playground/data/eval/pope/val2014 \\
            --answers-file ./playground/data/eval/pope/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl \\
            --temperature 0 \\
            --conv-mode vicuna_v1 \\
            --pruning_method ${METHOD} \\
            --visual_token_num ${TOKEN_NUM} \\
            --enable-dynamic-selection \\
            --min-heads ${MIN_HEADS} \\
            --max-heads ${MAX_HEADS}

        echo ""
        echo "📊 计算POPE分数..."
        python llava/eval/eval_pope.py \\
            --annotation-dir ./playground/data/eval/pope/coco \\
            --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \\
            --result-file ./playground/data/eval/pope/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl
    } 2>&1 | tee -a ${RESULT_FILE}
}

# MME评估
eval_mme() {
    local TOKEN_NUM=$1
    local RESULT_FILE="${WORK_DIR}/head_strategy_mme_dynamic_${TOKEN_NUM}_results.txt"

    echo "🎯 执行MME评估 (Token: ${TOKEN_NUM})"

    {
        echo "TOKEN=${TOKEN_NUM}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:"
        python -m llava.eval.model_vqa_mme \\
            --model-path ${CKPT_DIR}/${CKPT} \\
            --model-base lmsys/vicuna-7b-v1.5 \\
            --question-file ./playground/data/eval/MME/llava_mme.jsonl \\
            --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \\
            --answers-file ./playground/data/eval/MME/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl \\
            --temperature 0 \\
            --conv-mode vicuna_v1 \\
            --single-pred-prompt \\
            --pruning_method ${METHOD} \\
            --visual_token_num ${TOKEN_NUM} \\
            --enable-dynamic-selection \\
            --min-heads ${MIN_HEADS} \\
            --max-heads ${MAX_HEADS}

        echo ""
        echo "📊 计算MME分数..."
        cd ./playground/data/eval/MME
        python calculation.py --results_dir answers
        cd ${WORK_DIR}
    } 2>&1 | tee -a ${RESULT_FILE}
}

# TextVQA评估
eval_textvqa() {
    local TOKEN_NUM=$1
    local RESULT_FILE="${WORK_DIR}/head_strategy_textvqa_dynamic_${TOKEN_NUM}_results.txt"

    echo "🎯 执行TextVQA评估 (Token: ${TOKEN_NUM})"

    {
        echo "TOKEN=${TOKEN_NUM}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:"
        python -m llava.eval.model_vqa_loader \\
            --model-path ${CKPT_DIR}/${CKPT} \\
            --model-base lmsys/vicuna-7b-v1.5 \\
            --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \\
            --image-folder ./playground/data/eval/textvqa/train_images \\
            --answers-file ./playground/data/eval/textvqa/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl \\
            --temperature 0 \\
            --conv-mode vicuna_v1 \\
            --pruning_method ${METHOD} \\
            --visual_token_num ${TOKEN_NUM} \\
            --enable-dynamic-selection \\
            --min-heads ${MIN_HEADS} \\
            --max-heads ${MAX_HEADS}

        echo ""
        echo "📊 计算TextVQA分数..."
        python llava/eval/eval_textvqa.py \\
            --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \\
            --result-file ./playground/data/eval/textvqa/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl
    } 2>&1 | tee -a ${RESULT_FILE}
}

# GQA评估
eval_gqa() {
    local TOKEN_NUM=$1
    local RESULT_FILE="${WORK_DIR}/head_strategy_gqa_dynamic_${TOKEN_NUM}_results.txt"

    echo "🎯 执行GQA评估 (Token: ${TOKEN_NUM})"

    {
        echo "TOKEN=${TOKEN_NUM}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:"
        python -m llava.eval.model_vqa_loader \\
            --model-path ${CKPT_DIR}/${CKPT} \\
            --model-base lmsys/vicuna-7b-v1.5 \\
            --question-file ./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \\
            --image-folder ./playground/data/eval/gqa/images \\
            --answers-file ./playground/data/eval/gqa/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl \\
            --temperature 0 \\
            --conv-mode vicuna_v1 \\
            --pruning_method ${METHOD} \\
            --visual_token_num ${TOKEN_NUM} \\
            --enable-dynamic-selection \\
            --min-heads ${MIN_HEADS} \\
            --max-heads ${MAX_HEADS}

        echo ""
        echo "📊 计算GQA分数..."
        python llava/eval/eval_gqa.py \\
            --annotation-file ./playground/data/eval/gqa/testdev_balanced_questions.json \\
            --result-file ./playground/data/eval/gqa/answers/${CKPT}_${METHOD}_${TOKEN_NUM}.jsonl
    } 2>&1 | tee -a ${RESULT_FILE}
}

# ========== 主执行流程 ==========
main() {
    echo "🚀 开始动态Head选择评估"
    echo "📊 配置信息:"
    echo "   - Benchmark: ${BENCHMARK}"
    echo "   - Visual tokens: ${TOKEN_NUMS[*]}"
    echo "   - Head range: [${MIN_HEADS}, ${MAX_HEADS}]"
    echo "   - GPU配置: ${GPULIST} (${CHUNKS} chunks)"
    echo ""

    # 初始化
    setup_environment
    setup_gpu

    # 创建答案目录
    mkdir -p "./playground/data/eval/${BENCHMARK}/answers"

    # 为每个token数量运行评估
    for TOKEN_NUM in "${TOKEN_NUMS[@]}"; do
        echo ""
        echo "🎯 开始评估: ${BENCHMARK} with ${TOKEN_NUM} tokens"
        echo "=================================================="

        case $BENCHMARK in
            "pope")
                eval_pope $TOKEN_NUM
                ;;
            "mme")
                eval_mme $TOKEN_NUM
                ;;
            "textvqa")
                eval_textvqa $TOKEN_NUM
                ;;
            "gqa")
                eval_gqa $TOKEN_NUM
                ;;
            *)
                echo "❌ 未知benchmark: $BENCHMARK"
                exit 1
                ;;
        esac

        echo "✅ 完成: ${BENCHMARK} with ${TOKEN_NUM} tokens"
    done

    echo ""
    echo "🎉 所有评估完成!"
    echo "📁 结果文件: head_strategy_${BENCHMARK}_dynamic_*_results.txt"
}

# ========== 执行 ==========
main "$@"