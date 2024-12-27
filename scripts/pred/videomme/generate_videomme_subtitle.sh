#!/bin/bash

video_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/videos"
srt_path="/home/ligengz/nvr_elm_llm/dataset/Video-MME/subtitle"
output_path="/home/kaipoc/personal/research_vh/LLaVA-NeXT/playground/videomme/subtitle_txt"

python3 llava/eval/extract_videomme_subtitle.py \
    --video_path "${video_path}" \
    --srt_path "${srt_path}" \
    --num_frames 64 \
    --output_path "${output_path}"
