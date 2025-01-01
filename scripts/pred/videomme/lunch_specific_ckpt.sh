#!/bin/bash


use_subtitle="true"

exp_name="llava-qwen-7b"
model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
ckpt_iter="0"
output_name="pred_[${exp_name}]_[${ckpt_iter}]"
conv_model="qwen_1_5"

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/videomme/sbatch_single_ckpt.sh

if [ "${use_subtitle}" = "true" ] || [ "${use_subtitle}" = "True" ]; then
    sbatch --job-name=pred_${exp_name}_subtitles \
        --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/pred/videomme/with_subtitles/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
        --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/pred/videomme/with_subtitles/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
        /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/videomme/sbatch_single_ckpt.sh \
        "${model_path}" \
        "${output_name}" \
        "${conv_model}" \
        "${use_subtitle}"
else
    use_subtitle="false"
    sbatch --job-name=pred_${exp_name}_no_subtitles \
        --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/pred/videomme/without_subtitles/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
        --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/pred/videomme/without_subtitles/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
        /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/videomme/sbatch_single_ckpt.sh \
        "${model_path}" \
        "${output_name}" \
        "${conv_model}" \
        "${use_subtitle}"
fi