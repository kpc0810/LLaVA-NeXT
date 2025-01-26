#!/bin/bash

model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
data_dir="/home/kaipoc/personal/research_vh/LLaVA-NeXT/playground/FactVC/data"
output_dir="outputs/factvc/pred_results/llava-qwen-7b"
output_name="llava-qwen-7b"


torchrun --nnodes=1 --nproc_per_node=8 --master_port=51469 \
    --master_addr "127.0.0.1" --node_rank="0"  \
    llava/eval/model_caption_factvc.py \
    --model-path "${model_path}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "qwen_1_5" \
    --frames_upbound 64 \
    --data_dir "${data_dir}"