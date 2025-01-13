#!/bin/bash

exp_name=$1
which_iter=$2

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT
pred_data_dir="outputs/miradata/pred_results/${exp_name}"
score_data_dir="outputs/miradata/scores/${exp_name}"
score_file="score_[${exp_name}]_[${which_iter}].json"
pred_file="pred_[${exp_name}]_[${which_iter}].json"
echo "Running inference for $pred_file and save score to $score_file!"

chmod +x scripts/eval/miradata/sbatch_single_ckpt.sh
sbatch \
--job-name=em_${which_iter} \
--output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/eval/miradata/${exp_name}/${pred_file}.txt \
--error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/eval/miradata/${exp_name}/${pred_file}.txt \
scripts/eval/miradata/sbatch_single_ckpt.sh \
"$pred_data_dir" \
"$score_data_dir" \
"$pred_file" \
"$score_file"
