#!/bin/bash
huggingface_token="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"

which huggingface-cli
huggingface-cli login --token "$huggingface_token"

exp_name="llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=1e-5_vtlr=5e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.5_tpaclwt=0.5"
ckpt_iter=1000

pred_file="outputs/dream1k/pred_results/${exp_name}/pred_[${exp_name}]_[${ckpt_iter}].jsonl"
score_file="outputs/dream1k/scores/${exp_name}/score_[${exp_name}]_[${ckpt_iter}].json"
temp_data_dir="outputs/dream1k/temp_dir/${exp_name}"

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT

CUDA_VISIBLE_DEVICES=0 python3 llava/eval/evaluate_cap_llamaacc_dream1k.py \
    --pred_file "${pred_file}" \
    --score_file "${score_file}" \
    --dataset_name "DREAM-1K" \
    --temp_dir "${temp_data_dir}" \
    --debug