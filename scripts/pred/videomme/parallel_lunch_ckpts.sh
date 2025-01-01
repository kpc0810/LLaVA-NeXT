#!/bin/bash
exp_name=$1
max_concurrent_jobs=$2
conv_mode=$3
num_iterations=$4
use_subtitle=$5

work_dir=$(pwd)
echo "$work_dir"/checkpoints/exps/"$exp_name"/*/

if [[ "${use_subtitle}" = "true" || "${use_subtitle}" = "True" ]]; then
    log_dir="${work_dir}/slurm_log/pred/videomme/with_subtitles/${exp_name}"
    error_dir="${work_dir}/slurm_error/pred/videomme/with_subtitles/${exp_name}"
else
    log_dir="${work_dir}/slurm_log/pred/videomme/without_subtitles/${exp_name}"
    error_dir="${work_dir}/slurm_error/pred/videomme/without_subtitles/${exp_name}"
fi

all_ckpt_dir=$(ls -d "$work_dir"/checkpoints/exps/"$exp_name"/*/ | sort -V)

chmod +x scripts/pred/videomme/sbatch_single_ckpt.sh
# decide how many iterations to run for each ckpt
for ((i = 1; i <= num_iterations; i++)); do
    echo "Iteration $i/$num_iterations"

    # run each ckpt once in this iteration
    for single_ckpt_dir in $all_ckpt_dir; do
        if [[ $(basename "$single_ckpt_dir") == checkpoint* ]]; then
            ckpt_iter=$(basename "$single_ckpt_dir" | rev | cut -d'-' -f1 | rev)
            pred_filename="pred_[${exp_name}]_[${ckpt_iter}]"

            echo "Running inference for $single_ckpt_dir with output name $pred_filename!"

            sbatch \
            --job-name="${pred_filename}_videomme" \
            --output="${log_dir}/${pred_filename}.txt" \
            --error="${error_dir}/${pred_filename}.txt" \
            scripts/pred/videomme/sbatch_single_ckpt.sh "${single_ckpt_dir}" "${pred_filename}" "${conv_mode}" "${use_subtitle}"

        fi
    done
done