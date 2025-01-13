#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --partition=polar4,polar3,polar2,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --dependency=singleton
#SBATCH --export=ALL
#SBATCH --signal=B:SIGUSR1@90

source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava-eval
which conda
conda activate llava-eval

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

pred_data_dir=$1
score_data_dir=$2
pred_file=$3
score_file=$4
test_dataset_path="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/final_miradata_9k_test_dataset.csv"
word_embedding_model_path="/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/cc.en.300.bin"

n_node=${SLURM_JOB_NUM_NODES:-1}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}

echo "n_node: $n_node"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "CURRENT_RANK: $CURRENT_RANK"

function sig_handler_USR1()
{
    echo "=============================================================="
    echo "Signal trapped -  date"
    echo "Requeuing job $SLURM_JOB_ID for the $SLURM_RESTART_COUNT time."
    echo "=============================================================="
    # requeue job
    scontrol requeue $SLURM_JOB_ID
    exit 2
}

trap 'sig_handler_USR1' SIGUSR1

srun torchrun --nnodes="${n_node}" --nproc_per_node=8 --master_port=51466 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/evaluate_cap_hallu_miradata.py \
    --word_embedding_model_path "$word_embedding_model_path" \
    --pred_data_dir "$pred_data_dir" \
    --score_data_dir "$score_data_dir" \
    --test_dataset_path "$test_dataset_path" \
    --pred_file "$pred_file" \
    --score_file "$score_file" &

wait
