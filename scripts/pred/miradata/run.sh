#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava
which conda
which scontrol
which torchrun
conda activate llava-eval
which python

cd /home/kaipoc/personal/research_vh/LLaVA-NeXT/

# export to avoid NCCL error
export DECORD_DUPLICATE_WARNING_THRESHOLD=1.0
export OMP_NUM_THREADS=1
export NCCL_BLOCKING_WAIT=0
export NCCL_IB_SL=0
export NCCL_IB_TC=41
export NCCL_TIMEOUT_MS=7200000
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=31
export NCCL_SOCKET_IFNAME=eth1
export CUDA_DEVICE_MAX_CONNECTIONS=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export HYDRA_FULL_ERROR="1"
export DECORD_EOF_RETRY_MAX="100000"
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR: $MASTER_ADDR"
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"
echo "Number of nodes: $n_node"
echo "CURRENT_RANK: $CURRENT_RANK"


MODEL_PATH=$1
OUTPUT_NAME=$2
CONV_MODE=$3
frames_upbound=64

# Extract the content inside the first set of square brackets in OUTPUT_NAME
exp_name=$(echo $OUTPUT_NAME | grep -oP '(?<=\[)[^\]]+(?=\])' | head -n 1)
output_dir="outputs/miradata/pred_results/${exp_name}"

# fixed params
data_file="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/final_miradata_9k_test_dataset.csv"
video_folder="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/video/clip_video"

torchrun --nnodes="${n_node}" --nproc_per_node=8 --master_port=51466 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/model_caption_miradata.py \
    --model-path "${MODEL_PATH}" \
    --data-file "${data_file}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${OUTPUT_NAME}" \
    --conv-mode "${CONV_MODE}" \
    --frames_upbound "${frames_upbound}" \
    --mm_spatial_pool_stride "2" \
    --image_aspect_ratio "anyres" \
    --image_grid_pinpoints "[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]" \
    --mm_patch_merge_type "spatial_unpad" \
    --overwrite "true" \
    --for_get_frames_num "4" \
    --load_8bit "false"