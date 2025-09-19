#!/bin/bash

# =============================================================================
# 快速策略评估脚本 - 测试关键配置组合
# =============================================================================

# 基本配置
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"

# 定义要测试的聚合策略
AGGREGATION_STRATEGIES=("original" "quality_based" "entropy_based")

# 定义要测试的任务（选择代表性任务）
TASKS=("mme" "vqav2" "gqa")

# 定义要测试的配置（选择关键配置）
CONFIGS=(
    "192 16"  # 高token数，中等head数
    "128 16"  # 中等token数，中等head数
    "64 8"    # 低token数，低head数
)

# 创建结果目录
RESULT_DIR="./quick_evaluation_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# 日志文件
LOG_FILE="$RESULT_DIR/evaluation.log"

echo "==============================================================================" | tee -a "$LOG_FILE"
echo "开始快速策略评估: $(date)" | tee -a "$LOG_FILE"
echo "GPU列表: $gpu_list" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

# 函数：运行单个评估任务
run_quick_evaluation() {
    local task=$1
    local strategy=$2
    local token=$3
    local head=$4
    
    local method="fastv+finepruner+head"
    local param="vtn_${token}_${head}_${strategy}"
    
    echo "运行评估: 任务=${task}, 策略=${strategy}, token=${token}, head=${head}" | tee -a "$LOG_FILE"
    
    # 使用现有的脚本，添加聚合策略参数
    case $task in
        "mme")
            CUDA_VISIBLE_DEVICES=$gpu_list bash scripts/v1_5/7b/mme.sh "$method" "$token" "$head" "$strategy"
            ;;
        "vqav2")
            CUDA_VISIBLE_DEVICES=$gpu_list bash scripts/v1_5/7b/vqav2.sh "$method" "$token" "$head" "$strategy"
            ;;
        "gqa")
            CUDA_VISIBLE_DEVICES=$gpu_list bash scripts/v1_5/7b/gqa.sh "$method" "$token" "$head" "$strategy"
            ;;
        *)
            echo "未支持的任务: $task" | tee -a "$LOG_FILE"
            return 1
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
        for config in "${CONFIGS[@]}"; do
            ((total_tasks++))
        done
    done
done

echo "总计任务数: $total_tasks" | tee -a "$LOG_FILE"

# 执行评估
for task in "${TASKS[@]}"; do
    for strategy in "${AGGREGATION_STRATEGIES[@]}"; do
        for config in "${CONFIGS[@]}"; do
            # 解析配置
            read -r token head <<< "$config"
            
            echo "进度: $((completed_tasks + 1))/$total_tasks" | tee -a "$LOG_FILE"
            
            # 运行评估
            if run_quick_evaluation "$task" "$strategy" "$token" "$head"; then
                ((completed_tasks++))
                echo "成功完成: $completed_tasks/$total_tasks" | tee -a "$LOG_FILE"
            else
                echo "失败: 任务=${task}, 策略=${strategy}, token=${token}, head=${head}" | tee -a "$LOG_FILE"
            fi
            
            # 添加间隔
            sleep 1
        done
    done
done

echo "==============================================================================" | tee -a "$LOG_FILE"
echo "快速评估完成: $(date)" | tee -a "$LOG_FILE"
echo "成功完成: $completed_tasks/$total_tasks" | tee -a "$LOG_FILE"
echo "结果保存在: $RESULT_DIR" | tee -a "$LOG_FILE"
echo "==============================================================================" | tee -a "$LOG_FILE"

echo "快速策略评估脚本执行完成！"
