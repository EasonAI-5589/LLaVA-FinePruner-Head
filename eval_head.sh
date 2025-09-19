#!/bin/bash

# ========== Head策略消融研究脚本 ==========
echo "🧪 开始Head策略消融研究"

# 确保从正确的工作目录启动
WORK_DIR="/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/code/VTP/LLaVA-FinePruner-Head"
echo "切换到工作目录: ${WORK_DIR}"
cd "${WORK_DIR}" || { echo "❌ 错误: 无法切换到工作目录 ${WORK_DIR}"; exit 1; }

# 验证关键目录存在和结构
echo "验证关键目录..."
for dir in "playground/data/eval" "llava/eval"; do
    if [ ! -d "${dir}" ]; then
        echo "❌ 错误: 关键目录不存在: ${WORK_DIR}/${dir}"
        exit 1
    fi
done

# 检查并创建必要的子目录
echo "检查评估子目录..."
if [ ! -d "playground/data/eval/pope" ]; then
    echo "⚠️  警告: POPE目录不存在，正在创建..."
    mkdir -p playground/data/eval/pope
fi

if [ ! -d "playground/data/eval/MME" ]; then
    echo "⚠️  警告: MME目录不存在，正在创建..."
    mkdir -p playground/data/eval/MME
fi

# 检查是否存在错误的嵌套目录
if [ -d "playground/data/eval/playground" ]; then
    echo "⚠️  检测到错误的嵌套目录结构: playground/data/eval/playground"
    echo "建议手动清理此嵌套结构"
fi

echo "✅ 目录结构验证完成"

# 自动检测可用GPU数量
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
else
    GPU_COUNT=1
fi

# 生成GPU列表 (0,1,2,...)
gpu_list=$(seq -s, 0 $((GPU_COUNT-1)))
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "检测到 ${GPU_COUNT} 张GPU: ${gpu_list}"

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"

METHOD="ablation_a"

# 测试参数组合
TOKEN_NUMS=(192 128 64)
HEAD_NUMS=(24 16 8)

# 效果较好的头选择策略 + 新增复杂策略
GOOD_STRATEGIES=("max_attention" "attention_range" "sparsity" "top_k_sum" "multi_objective" "graph_based" "hierarchical")

# 创建总结果文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${WORK_DIR}/head_strategy_ablation_${TIMESTAMP}.txt"

echo "结果将保存到: ${SUMMARY_FILE}"
echo "Head策略消融研究评估结果" > $SUMMARY_FILE
echo "评估时间: $(date)" >> $SUMMARY_FILE
echo "方法: ${METHOD}" >> $SUMMARY_FILE
echo "测试Token数量: ${TOKEN_NUMS[@]}" >> $SUMMARY_FILE
echo "测试Head数量: ${HEAD_NUMS[@]}" >> $SUMMARY_FILE
echo "测试策略: ${GOOD_STRATEGIES[@]}" >> $SUMMARY_FILE
echo "===============================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# ========== POPE 评估 ==========
echo "📊 开始POPE评估"
POPE_SPLIT="llava_pope_test"

echo "POPE评估结果:" >> $SUMMARY_FILE
echo "=============" >> $SUMMARY_FILE

for TOKEN in "${TOKEN_NUMS[@]}"; do
    for HEAD in "${HEAD_NUMS[@]}"; do
        for strategy in "${GOOD_STRATEGIES[@]}"; do
            echo "🔍 评估配置: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${strategy}"

            PARAM="vtn_${TOKEN}_${HEAD}_${strategy}"

        # 创建输出目录
        mkdir -p ./playground/data/eval/pope/answers/${POPE_SPLIT}/${CKPT}/${METHOD}/${PARAM}

        # 并行运行多GPU
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                --model-path ${CKPT_DIR}/${CKPT} \
                --question-file ./playground/data/eval/pope/${POPE_SPLIT}.jsonl \
                --image-folder ${DATA_DIR}/pope/val2014 \
                --answers-file ./playground/data/eval/pope/answers/${POPE_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
                --num-chunks ${CHUNKS} \
                --chunk-idx ${IDX} \
                --pruning_method ${METHOD} \
                --visual_token_num ${TOKEN} \
                --H ${HEAD} \
                --head-selection-strategy ${STRATEGY} \
                --temperature 0 \
                --conv-mode vicuna_v1 &
        done

        wait

        # 合并结果
        output_file=./playground/data/eval/pope/answers/${POPE_SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl
        > "$output_file"

        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat ./playground/data/eval/pope/answers/${POPE_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
        done

            # 评估并保存结果
            echo "" >> $SUMMARY_FILE
            echo "TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${strategy}:" >> $SUMMARY_FILE
            python llava/eval/eval_pope.py \
                --annotation-dir ${DATA_DIR}/pope/coco \
                --question-file ./playground/data/eval/pope/${POPE_SPLIT}.jsonl \
                --result-file $output_file >> $SUMMARY_FILE
            echo "----------------------------------------" >> $SUMMARY_FILE

            echo "✅ POPE评估完成: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${strategy}"
        done
    done
done

echo "" >> $SUMMARY_FILE

# ========== MME 评估 ==========
echo "📊 开始MME评估"
MME_SPLIT="llava_mme"

echo "MME评估结果:" >> $SUMMARY_FILE
echo "============" >> $SUMMARY_FILE

for TOKEN in "${TOKEN_NUMS[@]}"; do
    for HEAD in "${HEAD_NUMS[@]}"; do
        for strategy in "${GOOD_STRATEGIES[@]}"; do
            echo "🔍 评估配置: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${strategy}"

            PARAM="vtn_${TOKEN}_${HEAD}_${strategy}"

            # 创建输出目录
            mkdir -p ./playground/data/eval/MME/answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM}

            # 并行运行多GPU
            for IDX in $(seq 0 $((CHUNKS-1))); do
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                    --model-path ${CKPT_DIR}/${CKPT} \
                    --question-file ./playground/data/eval/MME/${MME_SPLIT}.jsonl \
                    --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
                    --answers-file ./playground/data/eval/MME/answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
                    --num-chunks ${CHUNKS} \
                    --chunk-idx ${IDX} \
                    --pruning_method ${METHOD} \
                    --visual_token_num ${TOKEN} \
                    --H ${HEAD} \
                    --head-selection-strategy ${strategy} \
                    --temperature 0 \
                    --conv-mode vicuna_v1 &
            done

            wait

            # 合并结果
            output_file=./playground/data/eval/MME/answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl
            > "$output_file"

            for IDX in $(seq 0 $((CHUNKS-1))); do
                cat ./playground/data/eval/MME/answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
            done

            # 转换答案格式并评估
            echo "切换到MME评估目录进行后处理..."
            cd "${WORK_DIR}/playground/data/eval/MME" || { echo "错误: 无法切换到MME目录"; exit 1; }

            python convert_answer_to_mme.py \
                --data_path ${DATA_DIR}/MME \
                --experiment ${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge

            # 保存MME评估结果
            echo "" >> "${SUMMARY_FILE}"
            echo "TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${strategy}:" >> "${SUMMARY_FILE}"
            cd eval_tool || { echo "错误: 无法切换到eval_tool目录"; exit 1; }
            python calculation.py --results_dir answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM} >> "${SUMMARY_FILE}"
            echo "----------------------------------------" >> "${SUMMARY_FILE}"

            # 回到工作目录
            cd "${WORK_DIR}" || { echo "错误: 无法回到工作目录"; exit 1; }

            echo "✅ MME评估完成: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${strategy}"
        done
    done
done

echo "" >> $SUMMARY_FILE
echo "===============================================" >> $SUMMARY_FILE
echo "所有评估完成时间: $(date)" >> $SUMMARY_FILE

echo "✅ 所有Head策略消融研究评估完成!"
echo ""
echo "📊 评估结果已保存到: $SUMMARY_FILE"
echo "可以查看详细结果:"
echo "cat $SUMMARY_FILE"
echo ""
echo "📈 结果文件包含所有TOKEN、HEAD和STRATEGY组合的POPE和MME评估结果"