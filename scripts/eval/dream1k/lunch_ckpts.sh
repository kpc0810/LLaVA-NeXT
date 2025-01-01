#!/bin/bash

exp_name=$1

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT
work_dir=$(pwd)
pred_data_dir="outputs/dream1k/pred_results/${exp_name}"
score_data_dir="outputs/dream1k/scores/${exp_name}"
all_pred_files=$(ls -d "$work_dir"/"$pred_data_dir"/*.jsonl)

chmod +x scripts/eval/dream1k/sbatch_single_ckpt.sh
for pred_file in $all_pred_files; do
    score_file=$(basename "$pred_file" | sed 's/pred_/score_/' | sed 's/\.jsonl$/\.json/')
    pred_file=$(basename "$pred_file")
    echo "Running inference for $pred_file and save score to $score_file!"

    sbatch \
    --job-name=vdo_hallu_eval_dream1k \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/eval/dream1k/${exp_name}/${pred_file}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/eval/dream1k/${exp_name}/${pred_file}.txt \
    scripts/eval/dream1k/sbatch_single_ckpt.sh \
    "${pred_data_dir}/${pred_file}" \
    "${score_data_dir}/${score_file}"
done