#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --partition=polar4,polar3,polar2,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --dependency=singleton
#SBATCH --export=ALL,NCCL_BLOCKING_WAIT=1,NCCL_ASYNC_ERROR_HANDLING=1,NCCL_DEBUG=INFO,NCCL_P2P_DISABLE=1,NCCL_P2P_LEVEL=NVL,NCCL_TIMEOUT_MS=6000000

model_path=$1
output_name=$2
conv_mode=$3

chmod +x scripts/pred/miradata/run.sh
srun --label /bin/bash scripts/pred/miradata/run.sh \
    $model_path \
    $output_name \
    $conv_mode