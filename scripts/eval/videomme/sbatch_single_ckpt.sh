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
gt_file="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/qa_old_format.json"

srun python3 llava/eval/evaluate_videomme.py \
    --gt_file "${gt_file}" \
    --pred_file "${pred_file}" \
    --score_file "${score_file}"