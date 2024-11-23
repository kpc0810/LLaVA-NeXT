#!/bin/bash
CONFIG_FILE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

EXP_NAME=$(basename "$CONFIG_FILE" .ini)

# 讀取 config.ini
total_runs=$(grep '^total_runs' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
num_train_epochs=$(grep '^num_train_epochs' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
plm_lr=$(grep '^plm_lr' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
vt_lr=$(grep '^vt_lr' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')
bs_per_device=$(grep '^bs_per_device' "$CONFIG_FILE" | awk -F ' = ' '{print $2}')

chmod +x /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/sbatch_dehallu.sh
for i in $(seq 1 $total_runs); do
    sbatch \
    --job-name=${EXP_NAME} \
    --output=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/train/${EXP_NAME}/run_${i}.txt \
    --error=/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_error/train/${EXP_NAME}/run_${i}.txt \
    /home/kaipoc/personal/research_vh/LLaVA-NeXT/scripts/video/train/7b/sbatch_dehallu.sh \
    "${EXP_NAME}" \
    "${i}" \
    "${num_train_epochs}" \
    "${plm_lr}" \
    "${vt_lr}" \
    "${bs_per_device}"
done
