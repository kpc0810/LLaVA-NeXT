#!/bin/bash

video_path="/mnt/home/kaipoc/research_vh/LLaVA-NeXT/playground/videomme/videos"
srt_path="/mnt/home/kaipoc/research_vh/LLaVA-NeXT/playground/videomme/subtitle"
output_path="/mnt/home/kaipoc/research_vh/LLaVA-NeXT/playground/videomme/subtitle_txt_new"

python3 llava/eval/extract_videomme_subtitle.py \
    --video_path "${video_path}" \
    --srt_path "${srt_path}" \
    --num_frames 64 \
    --output_path "${output_path}"
