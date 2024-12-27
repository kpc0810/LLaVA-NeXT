#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --partition=polar4,polar3,polar2,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --export=ALL

source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
which conda
conda activate llava-eval

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT

pred_file=$1
score_file=$2
temp_dir=$3

python3 llava/eval/evaluate_cap_llamaacc_dream1k.py \
    --pred_file "$pred_file" \
    --score_file "$score_file" \
    --temp_dir "$temp_dir" \
    --dataset_name "DREAM-1K"