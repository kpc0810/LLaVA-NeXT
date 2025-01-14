#!/bin/bash
# source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
source activate llava
which python

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export HYDRA_FULL_ERROR="1"
export DECORD_EOF_RETRY_MAX="100000"
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

cd /mnt/home/kaipoc/research_vh/LLaVA-NeXT

n_node=1
nproc_per_node=2
data_path="playground/VidHal/vidhal/annotations.json"
video_folder="playground/VidHal/vidhal/videos"
output_dir="outputs/vidhal/pred_result"
output_name="test_0102_6_nocap"
model_path="lmms-lab/LLaVA-Video-7B-Qwen2"

# CUDA_VISIBLE_DEVICES=0 torchrun llava/eval/dream1k.py \
torchrun --nnodes="${n_node}" --nproc_per_node="${nproc_per_node}" --master_port=51466 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/model_vqa_vidhal.py \
    --model-path "${model_path}" \
    --data-file "${data_path}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "qwen_1_5" \
    --frames_upbound 64 \