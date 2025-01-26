#!/bin/bash
# This script is used for evaluating caption hallucination in miradata using the LLaVA-NeXT framework.
# Usage:
#   ./script_name.sh <pred_data_dir> <score_data_dir> <pred_file> <score_file>
#
# Arguments:
#   pred_data_dir: Path to the directory containing prediction data.
#   score_data_dir: Path to the directory to save scoring results.
#   pred_file: Name of the prediction file.
#   score_file: Name of the scoring file.
#
# Example: 
#   bash /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/factvc/lunch_specific_ckpt.sh "llava-qwen-7b" "lmms-lab/LLaVA-Video-7B-Qwen2" "0"
#   bash /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/factvc/lunch_specific_ckpt.sh "/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25/checkpoint-640" "llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25" "640"
# bash /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/factvc/lunch_specific_ckpt.sh "/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/llava-qwen-7b_gl_nnode=8_epo=1_plmlr=5e-5_vtlr=1e-6_bspd=1/checkpoint-80" "llava-qwen-7b_gl_nnode=8_epo=1_plmlr=5e-5_vtlr=1e-6_bspd=1" "80"

model_path=$1
ckpt_name=$2
ckpt_iter=$3
output_name="pred_[${ckpt_name}]_[${ckpt_iter}]"
conv_model="qwen_1_5"

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/factvc/sbatch_single_ckpt.sh
sbatch --job-name="pf_${ckpt_iter}" \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/pred/factvc/${ckpt_name}/pred_${ckpt_name}_${ckpt_iter}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/pred/factvc/${ckpt_name}/pred_${ckpt_name}_${ckpt_iter}.txt \
    /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/factvc/sbatch_single_ckpt.sh \
    "${model_path}" \
    "${output_name}" \
    "${conv_model}"