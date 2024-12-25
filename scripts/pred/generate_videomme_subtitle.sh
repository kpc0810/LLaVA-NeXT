#!/bin/bash

# video_path="playground/videomme/videos"
# srt_path="playground/videomme/subtitle"
# output_path="playground/videomme/subtitle_txt"
video_path=$1
srt_path=$2
output_path=$3

python3 llava/eval/extract_videomme_subtitle.py \
    --video_path "${video_path}" \
    --srt_path "${srt_path}" \
    --num_frames 64 \
    --output_path "${output_path}"
