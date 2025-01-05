#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --partition=polar4,polar3,polar2,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --dependency=singleton
#SBATCH --export=ALL

source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
which conda
conda activate llava-eval

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT

pred_data_dir=$1
score_data_dir=$2
pred_file=$3
score_file=$4
test_dataset_path="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/final_miradata_9k_test_dataset.csv"

python3 llava/eval/evaluate_cap_hallu_miradata.py \
    --pred_data_dir "$pred_data_dir" \
    --score_data_dir "$score_data_dir" \
    --test_dataset_path "$test_dataset_path" \
    --pred_file "$pred_file" \
    --score_file "$score_file"