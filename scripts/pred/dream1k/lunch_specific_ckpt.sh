#!/bin/bash

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/dream1k/sbatch_single_ckpt.sh

exp_name="llava-qwen-7b"
ckpt_name="lmms-lab/LLaVA-Video-7B-Qwen2"
ckpt_iter="0"
output_name="pred_[${exp_name}]_[${ckpt_iter}]"
conv_model="qwen_1_5"

sbatch --job-name=pred_${exp_name} \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/pred/dream1k/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/pred/dream1k/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
    /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/dream1k/sbatch_single_ckpt.sh \
    "${ckpt_name}" \
    "${output_name}" \
    "${conv_model}"
  