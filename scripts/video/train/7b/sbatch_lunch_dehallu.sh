#!/bin/bash
CONFIG_FILE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

EXP_NAME=$(basename "$CONFIG_FILE" .ini)

# read config.ini
total_runs=$(grep '^total_runs' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
n_nodes=$(grep '^n_nodes' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
num_train_epochs=$(grep '^num_train_epochs' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
plm_lr=$(grep '^plm_lr' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
vt_lr=$(grep '^vt_lr' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
bs_per_device=$(grep '^bs_per_device' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
dehallu_finetune=$(grep '^dehallu_finetune' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
cp_lr=$(grep '^cp_lr' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
as_lr=$(grep '^as_lr' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
vccl_wt=$(grep '^vccl_wt' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
tpocl_wt=$(grep '^tpocl_wt' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
tpacl_wt=$(grep '^tpacl_wt' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
use_hard_neg=$(grep '^use_hard_neg' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/sbatch_dehallu.sh
for i in $(seq 1 $total_runs); do
    sbatch \
    --job-name=tm \
    --nodes=${n_nodes} \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/train/${EXP_NAME}/run_${i}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/train/${EXP_NAME}/run_${i}.txt \
    /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/sbatch_dehallu.sh \
    "${EXP_NAME}" \
    "${i}" \
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
done
