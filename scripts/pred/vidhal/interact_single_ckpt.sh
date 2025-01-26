#!/bin/bash

model_path="/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25/checkpoint-640"
output_dir="outputs/vidhal/pred_results/llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"
output_name="pred_[llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25]_[640]"

# fixed params
data_path="playground/VidHal/vidhal/annotations.json"
video_folder="playground/VidHal/vidhal/videos"

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT

torchrun --nnodes=1 --nproc_per_node=8 --master_port=51469 \
    --master_addr "127.0.0.1" --node_rank="0"  \
    llava/eval/model_vqa_vidhal.py \
    --model-path "${model_path}" \
    --data-file "${data_path}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "qwen_1_5" \
    --frames_upbound 64 \
    --captioning