#!/bin/bash


exp_name="llava-qwen-7b_fcl_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.0"
ckpt_name="/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/llava-qwen-7b_fcl_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.0/checkpoint-320"
ckpt_iter="320"
output_name="pred_[${exp_name}]_[${ckpt_iter}].json"
conv_model="qwen_1_5"

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/miradata/sbatch_single_ckpt.sh
sbatch --job-name=pm \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/pred/miradata/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/pred/miradata/${exp_name}/pred_${exp_name}_${ckpt_iter}.txt \
    /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/pred/miradata/sbatch_single_ckpt.sh \
    "${ckpt_name}" \
    "${output_name}" \
    "${conv_model}"