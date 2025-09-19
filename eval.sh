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

# 配置参数
MODEL_PATH="./checkpoints/llava-v1.5-7b"
DATA_PATH="./playground/data/eval/pope/val.json"
IMAGE_FOLDER="./playground/data/eval/pope/val2014"
OUTPUT_DIR="./ablation_a_results"
H=16
VISUAL_TOKEN_NUM=128

# 头筛选策略列表
STRATEGIES=("sum" "variance" "entropy" "max_attention" "attention_range" "sparsity" "top_k_sum" "weighted_quality" "gini_coefficient")

mkdir -p "$OUTPUT_DIR"

# 运行所有头筛选策略
for strategy in "${STRATEGIES[@]}"; do
    echo "测试头筛选策略: $strategy"

    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --question-file "$DATA_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$OUTPUT_DIR/answers_$strategy.jsonl" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --pruning-method ablation_a \
        --H "$H" \
        --visual-token-num "$VISUAL_TOKEN_NUM" \
        --head-selection-strategy "$strategy"
done

echo "✅ 消融研究A实验完成! 结果保存在: $OUTPUT_DIR"