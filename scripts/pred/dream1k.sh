#!/bin/bash
source activate llava
which python

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export HYDRA_FULL_ERROR="1"
export DECORD_EOF_RETRY_MAX="100000"
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}
# n_node=1

cd /mnt/home/kaipoc/research_vh/LLaVA-NeXT

nproc_per_node=2
data_path="DREAM-1K/json/metadata.json"
video_folder="DREAM-1K/video/DREAM-1K_videos"
output_dir="outputs/dream1k"
output_name="dream1k_pred_results_"
question="Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."

# CUDA_VISIBLE_DEVICES=0 torchrun llava/eval/dream1k.py \
torchrun --nnodes="${n_node}" --nproc_per_node="${nproc_per_node}" --master_port=51466 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/dream1k.py \
    --model-path "lmms-lab/LLaVA-Video-7B-Qwen2" \
    --data-file "${data_path}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "qwen_1_5" \
    --frames_upbound 64 \
    --question "${question}" \

