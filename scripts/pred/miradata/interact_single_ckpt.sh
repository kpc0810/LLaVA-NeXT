#!/bin/bash

exp_name="llava-qwen-7b_gl_nnode=8_epo=1_plmlr=5e-6_vtlr=1e-6_bspd=1"
model_path="checkpoints/exps/llava-qwen-7b_gl_nnode=8_epo=1_plmlr=5e-6_vtlr=1e-6_bspd=1/checkpoint-360"
data_file="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/final_miradata_9k_test_dataset.csv"
video_folder="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/video/clip_video"
output_dir="outputs/miradata/pred_results/__debug"
output_name="__debug"
conv_mode="qwen_1_5"
frames_upbound="64"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=51466 \
    --master_addr "127.0.0.1" --node_rank="0"  \
    llava/eval/model_caption_miradata.py \
    --model-path "${model_path}" \
    --data-file "${data_file}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${output_name}" \
    --conv-mode "${conv_mode}" \
    --frames_upbound "${frames_upbound}" \
    --mm_spatial_pool_stride "2" \
    --image_aspect_ratio "anyres" \
    --image_grid_pinpoints "[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]" \
    --mm_patch_merge_type "spatial_unpad" \
    --overwrite "true" \
    --for_get_frames_num "4" \
    --load_8bit "false"