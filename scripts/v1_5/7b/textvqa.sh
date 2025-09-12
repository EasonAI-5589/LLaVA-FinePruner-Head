#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.5-7b"
SPLIT="llava_textvqa_val_v051_ocr"

METHOD=${1}
TOKEN=${2}
H=${3}
PARAM="vtn_${TOKEN}_${H}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/textvqa/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --H ${H} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_textvqa \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file
