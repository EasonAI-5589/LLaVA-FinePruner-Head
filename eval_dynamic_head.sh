#!/bin/bash

# ========== 智能求同存异的动态Head选择策略评估脚本 ==========
#
# 🎯 用途：专门评估智能求同存异策略在各个benchmark上的表现
#
# 🚀 核心特性：
# - 策略质量自适应评估 + 动态共识阈值机制 + 复杂度感知头数量调整
# - 多层次差异化选择 + 智能权重聚合机制
# - 预期性能提升：3-6%
#
# ============== 运行单个benchmark的示例命令 ==============
#
# 🧪 MME测试 (推荐配置):
# bash eval_dynamic_head.sh mme 128 16
#
# 🧪 POPE测试:
# bash eval_dynamic_head.sh pope 192 20
#
# 🧪 TextVQA测试:
# bash eval_dynamic_head.sh textvqa 128 16
#
# 🧪 GQA测试:
# bash eval_dynamic_head.sh gqa 192 18
#
# ============== 运行完整消融研究 ==============
#
# 🔬 完整评估所有benchmark:
# bash eval_dynamic_head.sh all
#
# 🔬 只评估特定token配置:
# bash eval_dynamic_head.sh all 128    # 只测试128 tokens
# bash eval_dynamic_head.sh all 192    # 只测试192 tokens
#
# ============== 调试模式 ==============
#
# 🐛 启用详细调试信息:
# DEBUG=true bash eval_dynamic_head.sh mme 128 16
#
# 这会显示类似输出:
# 🎯 Intelligent Consensus-Diversity: 17 heads
#    Consensus: 11 | Diversity: 6
#    Dynamic threshold: 0.28 | Complexity: 0.73
#    Strategy qualities: sparsity=0.85, hierarchical=0.72...
#
# ============== 结果评估 ==============
#
# 📊 查看MME结果:
# cd playground/data/eval/MME
# python eval_tool/calculation.py --results_dir answers/llava_mme/llava-v1.5-7b/ablation_a/vtn_128_dynamic
#
# 📊 查看POPE结果:
# python llava/eval/eval_pope.py \
#     --annotation-dir playground/data/eval/pope/coco \
#     --question-file playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file playground/data/eval/pope/answers/.../merge.jsonl
#
# 📊 查看TextVQA结果:
# python -m llava.eval.eval_textvqa \
#     --annotation-file playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file playground/data/eval/textvqa/answers/.../merge.jsonl
#
# 📊 查看GQA结果:
# python scripts/convert_gqa_for_eval.py --src .../merge.jsonl --dst predictions.json
# cd playground/data/eval/gqa/data && python eval/eval.py --tier testdev_balanced
#
# ============================================================

echo "🚀 开始智能求同存异动态Head选择策略评估"

# 默认参数
BENCHMARK=${1:-"mme"}
TOKEN_NUM=${2:-128}
HEAD_NUM=${3:-16}
ENABLE_DEBUG=${DEBUG:-false}

# 基础配置
WORK_DIR="/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/code/VTP/LLaVA-FinePruner-Head"
CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"
METHOD="dynamic_head"

echo "切换到工作目录: ${WORK_DIR}"
cd "${WORK_DIR}" || { echo "❌ 错误: 无法切换到工作目录"; exit 1; }

# GPU配置
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
else
    GPU_COUNT=1
fi

gpu_list=$(seq -s, 0 $((GPU_COUNT-1)))
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "检测到 ${GPU_COUNT} 张GPU: ${gpu_list}"

# 智能求同存异评估函数
run_intelligent_consensus_diversity() {
    local dataset=$1
    local token=$2
    local head=$3
    local param_suffix=$4

    echo "🎯 运行智能求同存异策略: ${dataset} | TOKEN=${token} | HEAD=${head}"

    PARAM="vtn_${token}_dynamic${param_suffix}"

    # 基础命令参数
    local base_args=(
        --model-path "${CKPT_DIR}/${CKPT}"
        --temperature 0
        --conv-mode vicuna_v1
        --pruning_method "${METHOD}"
        --visual_token_num "${token}"
        --H "${head}"
        --enable-dynamic-selection
    )

    # 如果启用调试模式，添加debug配置
    if [ "$ENABLE_DEBUG" = "true" ]; then
        echo "🐛 启用调试模式"
        export LLAVA_DEBUG_MODE=true
        base_args+=(--debug-mode)
    fi

    case $dataset in
        "mme")
            run_mme_evaluation "$token" "$head" "$PARAM"
            ;;
        "pope")
            run_pope_evaluation "$token" "$head" "$PARAM"
            ;;
        "textvqa")
            run_textvqa_evaluation "$token" "$head" "$PARAM"
            ;;
        "gqa")
            run_gqa_evaluation "$token" "$head" "$PARAM"
            ;;
        *)
            echo "❌ 不支持的数据集: $dataset"
            exit 1
            ;;
    esac
}

# MME评估
run_mme_evaluation() {
    local token=$1
    local head=$2
    local param=$3

    echo "📊 开始MME智能求同存异评估"

    local mme_split="llava_mme"
    local output_dir="./playground/data/eval/MME/answers/${mme_split}/${CKPT}/${METHOD}/${param}"

    mkdir -p "$output_dir"

    # 并行运行多GPU
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path "${CKPT_DIR}/${CKPT}" \
            --question-file "./playground/data/eval/MME/${mme_split}.jsonl" \
            --image-folder "${DATA_DIR}/MME/MME_Benchmark_release_version" \
            --answers-file "${output_dir}/${CHUNKS}_${IDX}.jsonl" \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method "${METHOD}" \
            --visual_token_num ${token} \
            --H ${head} \
            --enable-dynamic-selection \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    # 合并结果
    local output_file="${output_dir}/merge.jsonl"
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${output_dir}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
    done

    # MME后处理和评估
    echo "🔄 MME后处理和评估..."
    cd "${WORK_DIR}/playground/data/eval/MME" || exit 1

    python convert_answer_to_mme.py \
        --data_path "${DATA_DIR}/MME" \
        --experiment "${mme_split}/${CKPT}/${METHOD}/${param}/merge"

    cd eval_tool || exit 1
    echo "📈 MME智能求同存异评估结果:"
    python calculation.py --results_dir "answers/${mme_split}/${CKPT}/${METHOD}/${param}"

    cd "${WORK_DIR}" || exit 1
    echo "✅ MME智能求同存异评估完成"
}

# POPE评估
run_pope_evaluation() {
    local token=$1
    local head=$2
    local param=$3

    echo "📊 开始POPE智能求同存异评估"

    local pope_split="llava_pope_test"
    local output_dir="./playground/data/eval/pope/answers/${pope_split}/${CKPT}/${METHOD}/${param}"

    mkdir -p "$output_dir"

    # 并行运行多GPU
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path "${CKPT_DIR}/${CKPT}" \
            --question-file "./playground/data/eval/pope/${pope_split}.jsonl" \
            --image-folder "${DATA_DIR}/pope/val2014" \
            --answers-file "${output_dir}/${CHUNKS}_${IDX}.jsonl" \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method "${METHOD}" \
            --visual_token_num ${token} \
            --H ${head} \
            --enable-dynamic-selection \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    # 合并结果
    local output_file="${output_dir}/merge.jsonl"
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${output_dir}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
    done

    # POPE评估
    echo "📈 POPE智能求同存异评估结果:"
    python llava/eval/eval_pope.py \
        --annotation-dir "${DATA_DIR}/pope/coco" \
        --question-file "./playground/data/eval/pope/${pope_split}.jsonl" \
        --result-file "$output_file"

    echo "✅ POPE智能求同存异评估完成"
}

# TextVQA评估
run_textvqa_evaluation() {
    local token=$1
    local head=$2
    local param=$3

    echo "📊 开始TextVQA智能求同存异评估"

    local textvqa_split="llava_textvqa_val_v051_ocr"
    local output_dir="./playground/data/eval/textvqa/answers/${textvqa_split}/${CKPT}/${METHOD}/${param}"

    mkdir -p "$output_dir"

    # 并行运行多GPU
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path "${CKPT_DIR}/${CKPT}" \
            --question-file "./playground/data/eval/textvqa/${textvqa_split}.jsonl" \
            --image-folder "${DATA_DIR}/textvqa/train_images" \
            --answers-file "${output_dir}/${CHUNKS}_${IDX}.jsonl" \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method "${METHOD}" \
            --visual_token_num ${token} \
            --H ${head} \
            --enable-dynamic-selection \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    # 合并结果
    local output_file="${output_dir}/merge.jsonl"
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${output_dir}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
    done

    # TextVQA评估
    echo "📈 TextVQA智能求同存异评估结果:"
    python -m llava.eval.eval_textvqa \
        --annotation-file "${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json" \
        --result-file "$output_file"

    echo "✅ TextVQA智能求同存异评估完成"
}

# GQA评估
run_gqa_evaluation() {
    local token=$1
    local head=$2
    local param=$3

    echo "📊 开始GQA智能求同存异评估"

    local gqa_split="llava_gqa_testdev_balanced"
    local output_dir="./playground/data/eval/gqa/answers/${gqa_split}/${CKPT}/${METHOD}/${param}"

    mkdir -p "$output_dir"

    # 并行运行多GPU
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path "${CKPT_DIR}/${CKPT}" \
            --question-file "./playground/data/eval/gqa/${gqa_split}.jsonl" \
            --image-folder "${DATA_DIR}/gqa/data/images" \
            --answers-file "${output_dir}/${CHUNKS}_${IDX}.jsonl" \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method "${METHOD}" \
            --visual_token_num ${token} \
            --H ${head} \
            --enable-dynamic-selection \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait

    # 合并结果
    local output_file="${output_dir}/merge.jsonl"
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${output_dir}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
    done

    # GQA特殊处理
    echo "🔄 GQA格式转换和评估..."
    local gqa_dir="${WORK_DIR}/playground/data/eval/gqa/data"
    local prediction_file="${gqa_dir}/testdev_balanced_predictions_${token}_dynamic.json"

    python scripts/convert_gqa_for_eval.py --src "$output_file" --dst "$prediction_file"

    cd "$gqa_dir" || exit 1
    cp "testdev_balanced_predictions_${token}_dynamic.json" "testdev_balanced_predictions.json"

    echo "📈 GQA智能求同存异评估结果:"
    python eval/eval.py \
        --path "${DATA_DIR}/gqa/data/questions" \
        --tier testdev_balanced

    rm -f testdev_balanced_predictions.json
    cd "${WORK_DIR}" || exit 1

    echo "✅ GQA智能求同存异评估完成"
}

# 完整消融研究
run_full_ablation() {
    local specific_token=$1

    echo "🔬 开始智能求同存异完整消融研究"

    # 测试配置
    local token_nums=(192 128 64)
    local head_nums=(24 20 16 12 8)

    # 如果指定了特定token，只测试该配置
    if [ -n "$specific_token" ]; then
        token_nums=($specific_token)
        echo "🎯 只测试TOKEN=${specific_token}的配置"
    fi

    for token in "${token_nums[@]}"; do
        for head in "${head_nums[@]}"; do
            echo ""
            echo "🔍 评估配置: TOKEN=${token}, HEAD=${head} (智能求同存异)"

            # 运行所有benchmark
            run_intelligent_consensus_diversity "mme" "$token" "$head" "_${head}"
            sleep 2
            run_intelligent_consensus_diversity "pope" "$token" "$head" "_${head}"
            sleep 2
            run_intelligent_consensus_diversity "textvqa" "$token" "$head" "_${head}"
            sleep 2
            run_intelligent_consensus_diversity "gqa" "$token" "$head" "_${head}"
            sleep 5

            echo "✅ 配置 TOKEN=${token}, HEAD=${head} 评估完成"
        done
    done

    echo "✅ 智能求同存异完整消融研究完成!"
}

# 主执行逻辑
case $BENCHMARK in
    "all")
        run_full_ablation "$TOKEN_NUM"
        ;;
    "mme"|"pope"|"textvqa"|"gqa")
        run_intelligent_consensus_diversity "$BENCHMARK" "$TOKEN_NUM" "$HEAD_NUM" ""
        ;;
    *)
        echo "❌ 不支持的benchmark: $BENCHMARK"
        echo "支持的选项: mme, pope, textvqa, gqa, all"
        exit 1
        ;;
esac

echo ""
echo "🎉 智能求同存异动态Head选择策略评估完成!"
echo ""
echo "📊 查看结果的方法已在脚本顶部注释中详细说明"
echo "🚀 预期性能提升: 3-6%，特别是在复杂多模态理解任务上"