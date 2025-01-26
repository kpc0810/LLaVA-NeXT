#!/bin/bash
exp_name=$1
max_concurrent_jobs=$2
conv_mode=$3

work_dir=$(pwd)
echo "$work_dir"/checkpoints/exps/"$exp_name"/*/

all_ckpt_dir=$(ls -d "$work_dir"/checkpoints/exps/"$exp_name"/*/ | sort -V)

chmod +x scripts/pred/miradata/sbatch_single_ckpt.sh
# decide how many iterations to run for each ckpt
for single_ckpt_dir in $all_ckpt_dir; do
    if [[ $(basename "$single_ckpt_dir") == checkpoint* ]]; then
        ckpt_iter=$(basename "$single_ckpt_dir" | rev | cut -d'-' -f1 | rev)
        pred_filename="pred_[${exp_name}]_[${ckpt_iter}]"

        echo "Running inference for $single_ckpt_dir with output name $pred_filename!"

        sbatch \
        --job-name="pm" \
        --output="${work_dir}/slurm_log/pred/miradata/${exp_name}/${pred_filename}.txt" \
        --error="${work_dir}/slurm_error/pred/miradata/${exp_name}/${pred_filename}.txt" \
        scripts/pred/miradata/sbatch_single_ckpt.sh "${single_ckpt_dir}" "${pred_filename}" "${conv_mode}"

    fi
done