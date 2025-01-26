#!/bin/bash

exp_name=$1
which_iter=$2

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT
pred_data_dir="outputs/factvc/pred_results/${exp_name}"
score_data_dir="outputs/factvc/scores/${exp_name}"
score_file="score_[${exp_name}]_[${which_iter}].json"
pred_file="pred_[${exp_name}]_[${which_iter}].jsonl"
echo "Running inference for $pred_file and save score to $score_file!"

chmod +x scripts/eval/factvc/sbatch_single_ckpt.sh
sbatch \
    --job-name=ef_${which_iter} \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/eval/factvc/${exp_name}/${pred_file}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/eval/factvc/${exp_name}/${pred_file}.txt \
    scripts/eval/factvc/sbatch_single_ckpt.sh \
    "$pred_data_dir" \
    "$score_data_dir" \
    "$pred_file" \
    "$score_file"

