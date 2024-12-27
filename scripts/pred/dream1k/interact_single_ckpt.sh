#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
which python

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export HYDRA_FULL_ERROR="1"
export DECORD_EOF_RETRY_MAX="100000"
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT/

n_node=1
nproc_per_node=1
data_path="playground/DREAM-1K/json/metadata.json"
video_folder="playground/DREAM-1K/video/DREAM-1K_videos"
output_dir="outputs/dream1k/pred_results/__debug"
output_name="__debug"
model_path="checkpoints/exps/llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=1e-5_vtlr=5e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25/checkpoint-1000"
question="Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."

# CUDA_VISIBLE_DEVICES=0 torchrun llava/eval/dream1k.py \
torchrun --nnodes="${n_node}" --nproc_per_node="${nproc_per_node}" --master_port=51466 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/model_caption_dream1k.py \
    --model-path "${model_path}" \
    --data-file "${data_path}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "qwen_1_5" \
    --frames_upbound 64 \
    --question "${question}" \

