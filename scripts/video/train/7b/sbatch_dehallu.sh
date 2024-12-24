#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --partition=polar4,polar3,polar2,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --dependency=singleton
#SBATCH --exclusive

EXP_NAME=${1}
ROUND=${2}
num_train_epochs=${3}
plm_lr=${4}
vt_lr=${5}
bs_per_device=${6}
dehallu_finetune=${7}
cp_lr=${8}
as_lr=${9}
vccl_wt=${10}
tpocl_wt=${11}
tpacl_wt=${12}
use_hard_neg=${13}

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/dehallu.sh
srun --label bash /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/dehallu.sh \
    "/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/${EXP_NAME}" \
    "${EXP_NAME}_(${ROUND})" \
    "${num_train_epochs}" \
    "${plm_lr}" \
    "${vt_lr}" \
    "${bs_per_device}" \
    "${dehallu_finetune}" \
    "${cp_lr}" \
    "${as_lr}" \
    "${vccl_wt}" \
    "${tpocl_wt}" \
    "${tpacl_wt}" \
    "${use_hard_neg}"
