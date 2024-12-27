#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/miniconda3/bin/activate llava
which conda
which scontrol
which torchrun
conda activate llava
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


MODEL_PATH="checkpoints/exps/llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=1e-5_vtlr=5e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25/checkpoint-1000"
exp_name="llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=1e-5_vtlr=5e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"
output_dir="outputs/videomme/pred_results/__debug"
OUTPUT_NAME="__debug"
CONV_MODE="qwen_1_5"
frames_upbound=64

# fixed params
data_file="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/qa_old_format.json"
video_folder="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/videos"
subtitle_path="playground/videomme/subtitle_txt"


output_dir="outputs/videomme/with_subtitles/pred_results/${exp_name}"
torchrun --nnodes="${n_node}" --nproc_per_node=1 --master_port=51466 \
    --master_addr "${MASTER_ADDR}" --node_rank="${CURRENT_RANK}" \
    llava/eval/model_vqa_videomme.py \
    --model-path "${MODEL_PATH}" \
    --data-file "${data_file}" \
    --video_folder "${video_folder}" \
    --output-dir "${output_dir}" \
    --output-name "${OUTPUT_NAME}" \
    --conv-mode "${CONV_MODE}" \
    --frames_upbound "${frames_upbound}" \
    --use_subtitle \
    --subtitle_path "${subtitle_path}"