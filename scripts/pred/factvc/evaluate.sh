#!/bin/bash
# source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
source activate llava
which python

python3 llava/eval/evaluate_factvc.py \
    --pred_file $1  \
    --data_dir $2 \
    ${3:+ --save_dir $3} \
    --dataset "all"
