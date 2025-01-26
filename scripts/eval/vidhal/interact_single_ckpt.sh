# !/bin/bash

# params required to change
exp_name="llava-qwen-7b"
ckpt_iter=0

pred_data_dir="outputs/vidhal/pred_results/${exp_name}"
score_data_dir="outputs/vidhal/scores/${exp_name}"
pred_file="pred_[${exp_name}]_[${ckpt_iter}].jsonl"
score_file="score_[${exp_name}]_[${ckpt_iter}].json"
gt_file="playground/VidHal/vidhal/annotations.json"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=51466 \
    --master_addr "127.0.0.1" --node_rank="0"  \
    llava/eval/evaluate_vidhal.py \
    --gt_file "${gt_file}" \
    --pred_data_dir "${pred_data_dir}" \
    --score_data_dir "${score_data_dir}" \
    --pred_file "${pred_file}" \
    --score_file "${score_file}" \
    -