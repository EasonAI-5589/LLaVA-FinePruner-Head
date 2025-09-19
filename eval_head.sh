#!/bin/bash

# ========== Head数量和Token数量消融研究评估脚本 ==========
echo "🧪 开始Head数量和Token数量消融研究评估"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"

METHOD="fastv+finepruner"

# 测试参数组合
TOKEN_NUMS=(192 128 64)
HEAD_NUMS=(32 24 16 8)

# 创建总结果文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="./head_token_evaluation_summary_${TIMESTAMP}.txt"

echo "Head数量和Token数量消融研究评估结果" > $SUMMARY_FILE
echo "评估时间: $(date)" >> $SUMMARY_FILE
echo "方法: ${METHOD}" >> $SUMMARY_FILE
echo "测试Token数量: ${TOKEN_NUMS[@]}" >> $SUMMARY_FILE
echo "测试Head数量: ${HEAD_NUMS[@]}" >> $SUMMARY_FILE
echo "===============================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# ========== POPE 评估 ==========
echo "📊 开始POPE评估"
POPE_SPLIT="llava_pope_test"

echo "POPE评估结果:" >> $SUMMARY_FILE
echo "=============" >> $SUMMARY_FILE

for TOKEN in "${TOKEN_NUMS[@]}"; do
    for HEAD in "${HEAD_NUMS[@]}"; do
        echo "🔍 评估配置: TOKEN=${TOKEN}, HEAD=${HEAD}"

        PARAM="vtn_${TOKEN}_${HEAD}"

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
        echo "TOKEN=${TOKEN}, HEAD=${HEAD}:" >> $SUMMARY_FILE
        python llava/eval/eval_pope.py \
            --annotation-dir ${DATA_DIR}/pope/coco \
            --question-file ./playground/data/eval/pope/${POPE_SPLIT}.jsonl \
            --result-file $output_file >> $SUMMARY_FILE
        echo "----------------------------------------" >> $SUMMARY_FILE

        echo "✅ POPE评估完成: TOKEN=${TOKEN}, HEAD=${HEAD}"
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
        echo "🔍 评估配置: TOKEN=${TOKEN}, HEAD=${HEAD}"

        PARAM="vtn_${TOKEN}_${HEAD}"

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
        echo "TOKEN=${TOKEN}, HEAD=${HEAD}:" >> $SUMMARY_FILE
        cd eval_tool
        python calculation.py --results_dir answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM} >> $SUMMARY_FILE
        echo "----------------------------------------" >> $SUMMARY_FILE
        cd ../../../../

        echo "✅ MME评估完成: TOKEN=${TOKEN}, HEAD=${HEAD}"
    done
done

echo "" >> $SUMMARY_FILE
echo "===============================================" >> $SUMMARY_FILE
echo "所有评估完成时间: $(date)" >> $SUMMARY_FILE

echo "✅ 所有Head数量和Token数量消融研究评估完成!"
echo ""
echo "📊 评估结果已保存到: $SUMMARY_FILE"
echo "可以查看详细结果:"
echo "cat $SUMMARY_FILE"
echo ""
echo "📈 结果文件包含所有TOKEN和HEAD组合的POPE和MME评估结果"