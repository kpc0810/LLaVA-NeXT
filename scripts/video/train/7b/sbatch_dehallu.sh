#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=4:00:00
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --partition=polar4,polar3,polar2,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --dependency=singleton

EXP_NAME=${1}
ROUND=${2}
num_train_epochs=${3}
plm_lr=${4}
vt_lr=${5}
bs_per_device=${6}

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/dehallu.sh
srun --label bash /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/dehallu.sh \
    "/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/${EXP_NAME}" \
    "${EXP_NAME}_(${ROUND})" \
    "${num_train_epochs}" \
    "${plm_lr}" \
    "${vt_lr}" \
    "${bs_per_device}"
