#!/bin/bash

exp_name=$1
num_iterations=$2

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT
work_dir=$(pwd)
pred_data_dir="outputs/miradata/pred_results/${exp_name}"
score_data_dir="outputs/miradata/scores/${exp_name}"
all_pred_files=$(ls -d "$work_dir"/"$pred_data_dir"/*.json)

chmod +x scripts/eval/miradata/sbatch_single_ckpt.sh
for ((i = 1; i <= num_iterations; i++)); do
    echo "Iteration $i/$num_iterations"

    for pred_file in $all_pred_files; do
        score_file=$(basename "$pred_file" | sed 's/pred_/score_/')
        pred_file=$(basename "$pred_file")
        echo "Running inference for $pred_file and save score to $score_file!"

        sbatch \
        --job-name=em \
        --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/eval/miradata/${exp_name}/${pred_file}.txt \
        --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/eval/miradata/${exp_name}/${pred_file}.txt \
        scripts/eval/miradata/sbatch_single_ckpt.sh \
        "$pred_data_dir" \
        "$score_data_dir" \
        "$pred_file" \
        "$score_file"
    done
done