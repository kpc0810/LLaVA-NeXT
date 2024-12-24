#!/bin/bash
source activate llava
which python

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export HYDRA_FULL_ERROR="1"
export DECORD_EOF_RETRY_MAX="100000"
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
# n_node=${SLURM_JOB_NUM_NODES:-1}
n_node=1

cd /mnt/home/kaipoc/research_vh/LLaVA-NeXT

nproc_per_node=2
data_path="playground/videomme/qa_old_format.json"
video_folder="playground/videomme/videos"
output_dir="outputs/videomme"
output_name="videomme_sub_1224_5"
subtitle_path="playground/videomme/subtitle_txt"

torchrun --nnodes="${n_node}" --nproc_per_node="${nproc_per_node}" --master_port=51468 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/videomme.py \
    --model-path "lmms-lab/LLaVA-Video-7B-Qwen2" \
    --data-file "${data_path}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "qwen_1_5" \
    --frames_upbound 64 \
    --use_subtitle \
    --subtitle_path "${subtitle_path}"

python3 llava/eval/videomme_eval.py \
    --gt_file "${data_path}" \
    --pred_file "${output_dir}/${output_name}.jsonl"