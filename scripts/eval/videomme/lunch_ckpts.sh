#!/bin/bash

exp_name=$1
eval_subtitle=$2

if [[ "$eval_subtitle" = "true" || "$eval_subtitle" = "True" ]]; then
    pred_data_dir="outputs/videomme/with_subtitles/pred_results/${exp_name}"
    score_data_dir="outputs/videomme/with_subtitles/scores/${exp_name}"
    log_dir="slurm_log/eval/videomme/with_subtitles/${exp_name}"
    error_dir="slurm_error/eval/videomme/with_subtitles/${exp_name}"
else
    pred_data_dir="outputs/videomme/without_subtitles/pred_results/${exp_name}"
    score_data_dir="outputs/videomme/without_subtitles/scores/${exp_name}"
    log_dir="slurm_log/eval/videomme/without_subtitles/${exp_name}"
    error_dir="slurm_error/eval/videomme/without_subtitles/${exp_name}"
fi

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT
work_dir=$(pwd)
all_pred_files=$(ls -d "$work_dir"/"$pred_data_dir"/*.jsonl)

chmod +x scripts/eval/videomme/sbatch_single_ckpt.sh
for pred_file in $all_pred_files; do
    score_file=$(basename "$pred_file" | sed 's/pred_/score_/' | sed 's/\.jsonl$/\.json/')
    pred_file=$(basename "$pred_file")
    echo "Running inference for $pred_file and save score to $score_file!"

    sbatch \
    --job-name=vdo_hallu_eval_videomme \
    --output="${log_dir}/${pred_file}.txt" \
    --error="${error_dir}/${pred_file}.txt" \
    scripts/eval/videomme/sbatch_single_ckpt.sh \
    "${pred_data_dir}/${pred_file}" \
    "${score_data_dir}/${score_file}"
done