#!/bin/bash

# ========== Head策略消融研究脚本 ==========
echo "🧪 开始Head策略消融研究"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"

METHOD="ablation_a"

# 测试参数组合
TOKEN_NUMS=(192 128 64)
HEAD_NUMS=(24 16 8)

# 效果较好的头选择策略
GOOD_STRATEGIES=("max_attention" "attention_range" "sparsity" "top_k_sum")

# 创建总结果文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="./head_strategy_ablation_${TIMESTAMP}.txt"

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
        for STRATEGY in "${GOOD_STRATEGIES[@]}"; do
            echo "🔍 评估配置: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${STRATEGY}"

            PARAM="vtn_${TOKEN}_${HEAD}_${STRATEGY}"

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
            echo "TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${STRATEGY}:" >> $SUMMARY_FILE
            python llava/eval/eval_pope.py \
                --annotation-dir ${DATA_DIR}/pope/coco \
                --question-file ./playground/data/eval/pope/${POPE_SPLIT}.jsonl \
                --result-file $output_file >> $SUMMARY_FILE
            echo "----------------------------------------" >> $SUMMARY_FILE

            echo "✅ POPE评估完成: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${STRATEGY}"
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
        for STRATEGY in "${GOOD_STRATEGIES[@]}"; do
            echo "🔍 评估配置: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${STRATEGY}"

            PARAM="vtn_${TOKEN}_${HEAD}_${STRATEGY}"

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
                    --head-selection-strategy ${STRATEGY} \
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
            cd ./playground/data/eval/MME

            python convert_answer_to_mme.py \
                --data_path ${DATA_DIR}/MME \
                --experiment ${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge

            # 保存MME评估结果
            echo "" >> $SUMMARY_FILE
            echo "TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${STRATEGY}:" >> $SUMMARY_FILE
            cd eval_tool
            python calculation.py --results_dir answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM} >> $SUMMARY_FILE
            echo "----------------------------------------" >> $SUMMARY_FILE
            cd ../../../../

            echo "✅ MME评估完成: TOKEN=${TOKEN}, HEAD=${HEAD}, STRATEGY=${STRATEGY}"
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