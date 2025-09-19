#!/bin/bash

# =============================================================================
# 综合评估脚本 - 测试所有聚合策略
# =============================================================================

# 基本配置
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"

# 定义所有要测试的聚合策略
AGGREGATION_STRATEGIES=("original" "quality_based" "entropy_based" "topk_focus" "kurtosis_based")

# 定义要测试的任务
TASKS=("vqav2" "gqa" "vizwiz" "sqa" "textvqa" "pope" "mme" "mmbench" "mmbench_cn" "mmvet")

# 定义要测试的token数量
TOKENS=(192 128 64)

# 定义要测试的head数量  
HEADS=(32 24 16 8)

# 创建结果目录
RESULT_DIR="./evaluation_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 日志文件
LOG_FILE="$RESULT_DIR/evaluation.log"

# 记录开始时间
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "开始综合评估: $(date)" | tee -a "$LOG_FILE"
echo "GPU列表: $gpu_list" | tee -a "$LOG_FILE"
echo "并行chunks: $CHUNKS" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

# 函数：运行单个评估任务
run_evaluation() {
    local task=$1
    local strategy=$2
    local token=$3
    local head=$4
    
    local method="fastv+finepruner+head"
    local param="vtn_${token}_${head}_${strategy}"
    
    echo "运行评估: 任务=${task}, 策略=${strategy}, token=${token}, head=${head}" | tee -a "$LOG_FILE"
    
    # 创建输出目录
    local output_dir="./playground/data/eval/${task}/answers/llava_${task}/${CKPT}/${method}/${param}"
    mkdir -p "$output_dir"
    
    # 根据任务类型选择对应的脚本和数据文件
    case $task in
        "vqav2")
            local split="llava_vqav2_mscoco_test-dev2015"
            local question_file="./playground/data/eval/vqav2/${split}.jsonl"
            local image_folder="${DATA_DIR}/vqav2/test2015"
            ;;
        "gqa")
            local split="llava_gqa_testdev_balanced"
            local question_file="./playground/data/eval/gqa/${split}.jsonl"
            local image_folder="${DATA_DIR}/gqa/images"
            ;;
        "vizwiz")
            local split="llava_vizwiz_test"
            local question_file="./playground/data/eval/vizwiz/${split}.jsonl"
            local image_folder="${DATA_DIR}/vizwiz/test"
            ;;
        "sqa")
            local split="llava_sqa_test"
            local question_file="./playground/data/eval/scienceqa/${split}.json"
            local image_folder="${DATA_DIR}/scienceqa/images/test"
            ;;
        "textvqa")
            local split="llava_textvqa_val_v051_ocr"
            local question_file="./playground/data/eval/textvqa/${split}.jsonl"
            local image_folder="${DATA_DIR}/textvqa/train_images"
            ;;
        "pope")
            local split="llava_pope_test"
            local question_file="./playground/data/eval/pope/${split}.jsonl"
            local image_folder="${DATA_DIR}/coco/val2014"
            ;;
        "mme")
            local split="llava_mme"
            local question_file="./playground/data/eval/MME/${split}.jsonl"
            local image_folder="${DATA_DIR}/MME/MME_Benchmark_release_version"
            ;;
        "mmbench")
            local split="mmbench_dev_20230712"
            local question_file="./playground/data/eval/mmbench/${split}.tsv"
            local image_folder="${DATA_DIR}/mmbench/images"
            ;;
        "mmbench_cn")
            local split="mmbench_dev_cn_20231003"
            local question_file="./playground/data/eval/mmbench_cn/${split}.tsv"
            local image_folder="${DATA_DIR}/mmbench/images"
            ;;
        "mmvet")
            local split="llava_mmvet"
            local question_file="./playground/data/eval/mm-vet/${split}.jsonl"
            local image_folder="${DATA_DIR}/mm-vet/images"
            ;;
        *)
            echo "未知任务: $task" | tee -a "$LOG_FILE"
            return 1
            ;;
    esac
    
    # 并行运行评估
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file "$question_file" \
            --image-folder "$image_folder" \
            --answers-file "${output_dir}/${CHUNKS}_${IDX}.jsonl" \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --pruning_method ${method} \
            --visual_token_num ${token} \
            --H ${head} \
            --aggregation_strategy ${strategy} \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    
    wait
    
    # 合并结果
    local merge_file="${output_dir}/merge.jsonl"
    > "$merge_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${output_dir}/${CHUNKS}_${IDX}.jsonl" >> "$merge_file"
    done
    
    # 记录结果路径
    echo "结果保存至: $merge_file" | tee -a "$LOG_FILE"
    
    # 任务特定的后处理
    case $task in
        "vqav2")
            python scripts/convert_vqav2_for_submission.py \
                --dir "./playground/data/eval/vqav2" \
                --src "answers/llava_vqav2_mscoco_test-dev2015/${CKPT}/${method}/${param}/merge.jsonl" \
                --dst "answers_upload/llava_vqav2_mscoco_test-dev2015/${CKPT}/${method}/${param}.json"
            ;;
        "mme")
            cd ./playground/data/eval/MME
            python convert_answer_to_mme.py \
                --data_path ${DATA_DIR}/MME \
                --experiment "llava_mme/${CKPT}/${method}/${param}/merge"
            cd eval_tool
            python calculation.py --results_dir "answers/llava_mme/${CKPT}/${method}/${param}" | tee -a "$LOG_FILE"
            cd ../../../../..
            ;;
    esac
    
    echo "任务完成: 任务=${task}, 策略=${strategy}, token=${token}, head=${head}" | tee -a "$LOG_FILE"
}

# 主评估循环
total_tasks=0
completed_tasks=0

# 计算总任务数
for task in "${TASKS[@]}"; do
    for strategy in "${AGGREGATION_STRATEGIES[@]}"; do
        for token in "${TOKENS[@]}"; do
            for head in "${HEADS[@]}"; do
                ((total_tasks++))
            done
        done
    done
done

echo "总计任务数: $total_tasks" | tee -a "$LOG_FILE"

# 执行评估
for task in "${TASKS[@]}"; do
    for strategy in "${AGGREGATION_STRATEGIES[@]}"; do
        for token in "${TOKENS[@]}"; do
            for head in "${HEADS[@]}"; do
                echo "进度: $((completed_tasks + 1))/$total_tasks" | tee -a "$LOG_FILE"
                
                # 运行评估
                if run_evaluation "$task" "$strategy" "$token" "$head"; then
                    ((completed_tasks++))
                    echo "成功完成: $completed_tasks/$total_tasks" | tee -a "$LOG_FILE"
                else
                    echo "失败: 任务=${task}, 策略=${strategy}, token=${token}, head=${head}" | tee -a "$LOG_FILE"
                fi
                
                # 添加间隔以避免系统过载
                sleep 2
            done
        done
    done
done

# 生成评估报告
echo "==============================================================================" | tee -a "$LOG_FILE"
echo "评估完成: $(date)" | tee -a "$LOG_FILE"
echo "成功完成: $completed_tasks/$total_tasks" | tee -a "$LOG_FILE"
echo "结果保存在: $RESULT_DIR" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

# 创建结果摘要
python -c "
import os
import json
from datetime import datetime

# 创建摘要报告
summary = {
    'evaluation_date': '$(date)',
    'total_tasks': $total_tasks,
    'completed_tasks': $completed_tasks,
    'success_rate': round($completed_tasks / $total_tasks * 100, 2),
    'strategies_tested': $(printf '%s\n' "${AGGREGATION_STRATEGIES[@]}" | jq -R . | jq -s .),
    'tasks_tested': $(printf '%s\n' "${TASKS[@]}" | jq -R . | jq -s .),
    'tokens_tested': $(printf '%s\n' "${TOKENS[@]}" | jq -R . | jq -s .),
    'heads_tested': $(printf '%s\n' "${HEADS[@]}" | jq -R . | jq -s .),
    'result_directory': '$RESULT_DIR'
}

with open('$RESULT_DIR/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
    
print('评估摘要已保存至: $RESULT_DIR/summary.json')
"

echo "综合评估脚本执行完成！"
