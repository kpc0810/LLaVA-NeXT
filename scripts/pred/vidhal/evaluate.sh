#!/bin/bash
# source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
source activate llava
which python

python3 llava/eval/evaluate_vidhal.py \
    --pred_file $1  \
    --gt_file $2 \
    ${3:+ --score_file $3}
