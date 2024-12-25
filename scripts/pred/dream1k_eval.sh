#!/bin/bash
huggingface_token=$3

export PATH=/mnt/home/kaipoc/miniconda3/envs/llava/bin:$PATH
which huggingface-cli
huggingface-cli login --token "$huggingface_token"

python3 llava/eval/dream1k_llama_eval.py \
    --pred_path "$1" \
    --output_dir "$2" \
    --dataset_name "DREAM-1K"
