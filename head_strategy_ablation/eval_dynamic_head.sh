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

# 结果文件路径
POPE_RESULTS_FILE="${WORK_DIR}/head_strategy_pope_dynamic_results.txt"
MME_RESULTS_FILE="${WORK_DIR}/head_strategy_mme_dynamic_results.txt"
TEXTVQA_RESULTS_FILE="${WORK_DIR}/head_strategy_textvqa_dynamic_results.txt"
GQA_RESULTS_FILE="${WORK_DIR}/head_strategy_gqa_dynamic_results.txt"

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

    # 创建评估子目录
    echo "检查评估子目录..."
    for subdir in "pope" "MME" "gqa" "textvqa"; do
        [ ! -d "playground/data/eval/${subdir}" ] && {
            echo "⚠️  创建目录: playground/data/eval/${subdir}"
            mkdir -p "playground/data/eval/${subdir}"
        }
    done

    echo "✅ 目录结构验证完成"
}

# ========== GPU配置 ==========
setup_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        gpu_list=$(seq -s, 0 $((GPU_COUNT-1)))
        IFS=',' read -ra GPULIST <<< "$gpu_list"
        CHUNKS=${#GPULIST[@]}
    else
        GPU_COUNT=1
        GPULIST=(0)
        CHUNKS=1
        gpu_list="0"
    fi
    echo "检测到 ${GPU_COUNT} 张GPU: ${gpu_list}"
}

# ========== POPE 评估 ==========
eval_pope() {
    echo "📊 开始POPE评估"
    POPE_SPLIT="llava_pope_test"

    # 初始化POPE结果文件
    echo "动态Head策略POPE评估结果" > $POPE_RESULTS_FILE
    echo "评估时间: $(date)" >> $POPE_RESULTS_FILE
    echo "方法: ${METHOD}" >> $POPE_RESULTS_FILE
    echo "测试Token数量: ${TOKEN_NUMS[@]}" >> $POPE_RESULTS_FILE
    echo "Head范围: [${MIN_HEADS}, ${MAX_HEADS}]" >> $POPE_RESULTS_FILE
    echo "=============================================" >> $POPE_RESULTS_FILE
    echo "" >> $POPE_RESULTS_FILE

    for TOKEN in "${TOKEN_NUMS[@]}"; do
        echo "🔍 评估配置: TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}"

        PARAM="vtn_${TOKEN}_dynamic_${MIN_HEADS}_${MAX_HEADS}"

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
                --enable-dynamic-selection \
                --min-heads ${MIN_HEADS} \
                --max-heads ${MAX_HEADS} \
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
        echo "" >> $POPE_RESULTS_FILE
        echo "TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:" >> $POPE_RESULTS_FILE
        python llava/eval/eval_pope.py \
            --annotation-dir ${DATA_DIR}/pope/coco \
            --question-file ./playground/data/eval/pope/${POPE_SPLIT}.jsonl \
            --result-file $output_file >> $POPE_RESULTS_FILE
        echo "----------------------------------------" >> $POPE_RESULTS_FILE

        echo "✅ POPE评估完成: TOKEN=${TOKEN}"
    done
}

# ========== MME 评估 ==========
eval_mme() {
    echo "📊 开始MME评估"
    MME_SPLIT="llava_mme"

    # 初始化MME结果文件
    echo "动态Head策略MME评估结果" > $MME_RESULTS_FILE
    echo "评估时间: $(date)" >> $MME_RESULTS_FILE
    echo "方法: ${METHOD}" >> $MME_RESULTS_FILE
    echo "测试Token数量: ${TOKEN_NUMS[@]}" >> $MME_RESULTS_FILE
    echo "Head范围: [${MIN_HEADS}, ${MAX_HEADS}]" >> $MME_RESULTS_FILE
    echo "=============================================" >> $MME_RESULTS_FILE
    echo "" >> $MME_RESULTS_FILE

    for TOKEN in "${TOKEN_NUMS[@]}"; do
        echo "🔍 评估配置: TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}"

        PARAM="vtn_${TOKEN}_dynamic_${MIN_HEADS}_${MAX_HEADS}"

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
                --enable-dynamic-selection \
                --min-heads ${MIN_HEADS} \
                --max-heads ${MAX_HEADS} \
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
        echo "" >> "${MME_RESULTS_FILE}"
        echo "TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:" >> "${MME_RESULTS_FILE}"
        cd eval_tool || { echo "错误: 无法切换到eval_tool目录"; exit 1; }
        python calculation.py --results_dir answers/${MME_SPLIT}/${CKPT}/${METHOD}/${PARAM} >> "${MME_RESULTS_FILE}"
        echo "----------------------------------------" >> "${MME_RESULTS_FILE}"

        # 回到工作目录
        cd "${WORK_DIR}" || { echo "错误: 无法回到工作目录"; exit 1; }

        echo "✅ MME评估完成: TOKEN=${TOKEN}"
    done
}

# ========== TextVQA 评估 ==========
eval_textvqa() {
    echo "📊 开始TextVQA评估"
    TEXTVQA_SPLIT="llava_textvqa_val_v051_ocr"

    # 初始化TextVQA结果文件
    echo "动态Head策略TextVQA评估结果" > $TEXTVQA_RESULTS_FILE
    echo "评估时间: $(date)" >> $TEXTVQA_RESULTS_FILE
    echo "方法: ${METHOD}" >> $TEXTVQA_RESULTS_FILE
    echo "测试Token数量: ${TOKEN_NUMS[@]}" >> $TEXTVQA_RESULTS_FILE
    echo "Head范围: [${MIN_HEADS}, ${MAX_HEADS}]" >> $TEXTVQA_RESULTS_FILE
    echo "=============================================" >> $TEXTVQA_RESULTS_FILE
    echo "" >> $TEXTVQA_RESULTS_FILE

    for TOKEN in "${TOKEN_NUMS[@]}"; do
        echo "🔍 评估配置: TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}"

        PARAM="vtn_${TOKEN}_dynamic_${MIN_HEADS}_${MAX_HEADS}"

        # 创建输出目录
        mkdir -p ./playground/data/eval/textvqa/answers/${TEXTVQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}

        # 并行运行多GPU
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                --model-path ${CKPT_DIR}/${CKPT} \
                --question-file ./playground/data/eval/textvqa/${TEXTVQA_SPLIT}.jsonl \
                --image-folder ${DATA_DIR}/textvqa/train_images \
                --answers-file ./playground/data/eval/textvqa/answers/${TEXTVQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
                --num-chunks ${CHUNKS} \
                --chunk-idx ${IDX} \
                --pruning_method ${METHOD} \
                --visual_token_num ${TOKEN} \
                --enable-dynamic-selection \
                --min-heads ${MIN_HEADS} \
                --max-heads ${MAX_HEADS} \
                --temperature 0 \
                --conv-mode vicuna_v1 &
        done

        wait

        # 合并结果
        output_file=./playground/data/eval/textvqa/answers/${TEXTVQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl
        > "$output_file"

        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat ./playground/data/eval/textvqa/answers/${TEXTVQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
        done

        # 评估并保存结果
        echo "" >> $TEXTVQA_RESULTS_FILE
        echo "TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:" >> $TEXTVQA_RESULTS_FILE
        python -m llava.eval.eval_textvqa \
            --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
            --result-file $output_file >> $TEXTVQA_RESULTS_FILE
        echo "----------------------------------------" >> $TEXTVQA_RESULTS_FILE

        echo "✅ TextVQA评估完成: TOKEN=${TOKEN}"
    done
}

# ========== GQA 评估 ==========
eval_gqa() {
    echo "📊 开始GQA评估"
    GQA_SPLIT="llava_gqa_testdev_balanced"

    # 初始化GQA结果文件
    echo "动态Head策略GQA评估结果" > $GQA_RESULTS_FILE
    echo "评估时间: $(date)" >> $GQA_RESULTS_FILE
    echo "方法: ${METHOD}" >> $GQA_RESULTS_FILE
    echo "测试Token数量: ${TOKEN_NUMS[@]}" >> $GQA_RESULTS_FILE
    echo "Head范围: [${MIN_HEADS}, ${MAX_HEADS}]" >> $GQA_RESULTS_FILE
    echo "=============================================" >> $GQA_RESULTS_FILE
    echo "" >> $GQA_RESULTS_FILE

    for TOKEN in "${TOKEN_NUMS[@]}"; do
        echo "🔍 评估配置: TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}"

        PARAM="vtn_${TOKEN}_dynamic_${MIN_HEADS}_${MAX_HEADS}"

        # 创建输出目录
        mkdir -p ./playground/data/eval/gqa/answers/${GQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}

        # 并行运行多GPU
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                --model-path ${CKPT_DIR}/${CKPT} \
                --question-file ./playground/data/eval/gqa/${GQA_SPLIT}.jsonl \
                --image-folder ${DATA_DIR}/gqa/data/images \
                --answers-file ./playground/data/eval/gqa/answers/${GQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
                --num-chunks ${CHUNKS} \
                --chunk-idx ${IDX} \
                --pruning_method ${METHOD} \
                --visual_token_num ${TOKEN} \
                --enable-dynamic-selection \
                --min-heads ${MIN_HEADS} \
                --max-heads ${MAX_HEADS} \
                --temperature 0 \
                --conv-mode vicuna_v1 &
        done

        wait

        # 合并结果
        output_file=./playground/data/eval/gqa/answers/${GQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl
        > "$output_file"

        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat ./playground/data/eval/gqa/answers/${GQA_SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
        done

        # GQA特殊评估流程
        echo "进行GQA格式转换和评估..."
        GQA_DIR="${WORK_DIR}/playground/data/eval/gqa/data"
        prediction_file="${GQA_DIR}/testdev_balanced_predictions_${TOKEN}_dynamic_${MIN_HEADS}_${MAX_HEADS}.json"

        # 转换格式
        python scripts/convert_gqa_for_eval.py --src $output_file --dst $prediction_file

        # 切换到GQA评估目录
        cd ${GQA_DIR} || { echo "错误: 无法切换到GQA目录"; exit 1; }

        # 复制预测文件为评估程序期望的文件名
        cp "testdev_balanced_predictions_${TOKEN}_dynamic_${MIN_HEADS}_${MAX_HEADS}.json" "testdev_balanced_predictions.json"

        # 评估并保存结果
        echo "" >> $GQA_RESULTS_FILE
        echo "TOKEN=${TOKEN}, METHOD=${METHOD}, MIN_HEADS=${MIN_HEADS}, MAX_HEADS=${MAX_HEADS}:" >> $GQA_RESULTS_FILE
        python eval/eval.py \
            --path ${DATA_DIR}/gqa/data/questions \
            --tier testdev_balanced >> $GQA_RESULTS_FILE
        echo "----------------------------------------" >> $GQA_RESULTS_FILE

        # 清理临时文件
        rm -f testdev_balanced_predictions.json

        # 回到工作目录
        cd "${WORK_DIR}" || { echo "错误: 无法回到工作目录"; exit 1; }

        echo "✅ GQA评估完成: TOKEN=${TOKEN}"
    done
}

# ========== 主执行流程 ==========
main() {
    echo "🚀 开始动态Head选择评估"
    echo "📊 配置信息:"
    echo "   - Benchmark: ${BENCHMARK}"
    echo "   - Visual tokens: ${TOKEN_NUMS[*]}"
    echo "   - Head range: [${MIN_HEADS}, ${MAX_HEADS}]"
    echo ""

    # 初始化
    setup_environment
    setup_gpu

    echo "结果将分别保存到:"
    echo "POPE: ${POPE_RESULTS_FILE}"
    echo "MME: ${MME_RESULTS_FILE}"
    echo "TextVQA: ${TEXTVQA_RESULTS_FILE}"
    echo "GQA: ${GQA_RESULTS_FILE}"

    # 根据benchmark运行对应评估
    case $BENCHMARK in
        "pope")
            eval_pope
            ;;
        "mme")
            eval_mme
            ;;
        "textvqa")
            eval_textvqa
            ;;
        "gqa")
            eval_gqa
            ;;
        *)
            echo "❌ 未知benchmark: $BENCHMARK"
            exit 1
            ;;
    esac

    echo ""
    echo "✅ 所有动态Head策略评估完成!"
    echo ""
    echo "📊 评估结果已分别保存到:"
    echo "POPE结果: $POPE_RESULTS_FILE"
    echo "MME结果: $MME_RESULTS_FILE"
    echo "TextVQA结果: $TEXTVQA_RESULTS_FILE"
    echo "GQA结果: $GQA_RESULTS_FILE"
    echo ""
    echo "可以查看详细结果:"
    echo "cat $POPE_RESULTS_FILE"
    echo "cat $MME_RESULTS_FILE"
    echo "cat $TEXTVQA_RESULTS_FILE"
    echo "cat $GQA_RESULTS_FILE"
    echo ""
    echo "📈 每个文件包含所有TOKEN配置的对应评估结果"
}

# ========== 执行 ==========
main "$@"