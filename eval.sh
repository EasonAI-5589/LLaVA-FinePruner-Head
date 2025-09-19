cd# LLaVA Baseline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vqav2.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/gqa.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vizwiz.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/sqa.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench_cn.sh vanilla 576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmvet.sh vanilla 576

# FastV



# FastV+Finepruner

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vqav2.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vizwiz.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/textvqa.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/sqa.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/gqa.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmbench_cn.sh fastv+finepruner 192

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mmvet.sh fastv+finepruner 192

for task in pope mme vqav2 vizwiz sqa textvqa gqa mmbench mmbench_cn mmvet; do
    for token in 128 64; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash scripts/v1_5/7b/${task}.sh pdrop $token
    done
done


for task in vqav2 vizwiz mmbench mmbench_cn mmvet; do
    for token in 192 128 64; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash scripts/v1_5/7b/${task}.sh pdrop $token 32
    done
done

for task in vizwiz; do
    for token in 192 128 64; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash scripts/v1_5/7b/${task}.sh pdrop $token 32
    done
done


for task in sqa; do
    for token in 192 128 64; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash scripts/v1_5/7b/${task}.sh sparsevlm $token 32
    done
done


CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/textvqa.sh fastv+finepruner 192 16

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/gqa.sh fastv+finepruner 192 16

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/sqa.sh fastv+finepruner 192 16

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/mme.sh fastv+finepruner 128 8

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/mme.sh fastv+finepruner 128 16

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/mme.sh fastv+finepruner 192 8

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/mme.sh fastv+finepruner 128 24


CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/pope.sh fastv+finepruner 64 8

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/pope.sh fastv+finepruner 128 8



CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/gqa.sh fastv+finepruner 192 8

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/gqa.sh fastv+finepruner 128 24


for task in mmbench mmbench_cn; do
    for token in 128 64; do
        for head in 24 16 8; do
            CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
        done
    done
done



for task in mmvet; do
  for token in 192 128 64; do
    for head in 24 16 8; do
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
    done
  done
done

for task in mmbench mmbench_cn; do
  for token in 192 128 64; do
    for head in 24 16 8; do
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
    done
  done
done

for task in vqav2; do
  for token in 192; do
    for head in 8; do
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
    done
  done
done


for task in vqav2; do
  for token in 64; do
    for head in 16 8; do
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
    done
  done
done


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vqav2.sh fastv+finepruner 128 8


for task in textvqa; do
    for token in 192; do
        for head in 8; do
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
        done
    done
done



# # 运行脚本
# export PATH=$PATH:/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/env/anaconda3-2024.10/bin
# source activate
# conda activate star

for task in textvqa; do
    for token in 192; do
        for head in 8; do
            CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/7b/${task}.sh fastv+finepruner "$token" "$head"
        done
    done
done




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh fastv+finepruner 192 24

# ========== 消融研究A: 头筛选策略对比 ==========
echo "🧪 开始消融研究A: 头筛选策略对比实验"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"
CKPT="llava-v1.5-7b"
SPLIT="llava_pope_test"

# 消融研究A参数
METHOD="ablation_a"
TOKEN=128
HEAD=16

# 头筛选策略列表
STRATEGIES=("sum" "variance" "entropy" "max_attention" "attention_range" "sparsity" "top_k_sum" "weighted_quality" "gini_coefficient")

# 初始化结果汇总文件
echo "消融研究A: 头筛选策略POPE评估结果" > ./ablation_a_pope_results.txt
echo "测试时间: $(date)" >> ./ablation_a_pope_results.txt
echo "参数: TOKEN=${TOKEN}, HEAD=${HEAD}" >> ./ablation_a_pope_results.txt
echo "========================================" >> ./ablation_a_pope_results.txt
echo "" >> ./ablation_a_pope_results.txt

# 运行所有头筛选策略
for strategy in "${STRATEGIES[@]}"; do
    echo "测试头筛选策略: $strategy"

    PARAM="vtn_${TOKEN}_${HEAD}_${strategy}"

    # 创建输出目录
    mkdir -p ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}

    # 并行运行多GPU
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path ${CKPT_DIR}/${CKPT} \
            --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
            --image-folder ${DATA_DIR}/pope/val2014 \
            --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
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
    output_file=./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl
    > "$output_file"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # 评估结果
    echo "策略: $strategy" >> ./ablation_a_pope_results.txt
    python llava/eval/eval_pope.py \
        --annotation-dir ${DATA_DIR}/pope/coco \
        --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
        --result-file $output_file >> ./ablation_a_pope_results.txt
    echo "----------------------------------------" >> ./ablation_a_pope_results.txt

    echo "✅ 策略 $strategy 完成!"
done

echo "✅ 消融研究A实验完成!"
echo ""
echo "📊 所有策略评估结果已汇总到: ./ablation_a_pope_results.txt"
echo "可以直接查看比较各策略效果:"
echo "cat ./ablation_a_pope_results.txt"