#!/bin/bash

# set up wandb
export PYTHONWARNINGS="ignore"
export ACCELERATE_DEBUG_MODE="1"
export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER="1"


############### Prepare Envs #################
cd /home/kaipoc/personal/research_vh/LLaVA-NeXT/
alias python=python3

############### Show Envs ####################
ibstat
ibv_devinfo
nvidia-smi
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

port=29999
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export HYDRA_FULL_ERROR="1"
export DECORD_EOF_RETRY_MAX="100000"

################ Fixed Parameters ################
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION="qwen_1_5"
run_name="llavavideo-debug"
PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

################ Noticed Parameters ##############
n_node=1
nproc_per_node=8
output_dir="/home/kaipoc/personal/research_vh/LLaVA-NeXT/checkpoints/exps/debug"
dehallu_finetune=True
cp_lr=1e-4
vccl_wt=1.0
tpocl_wt=1.0
tpacl_wt=1.0
use_hard_neg=True

echo "master ip: ${master_addr}"
echo "master port: ${port}"

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
torchrun --nproc_per_node="${nproc_per_node}" --nnodes="${n_node}" --node_rank="${CURRENT_RANK}" --master_addr="${master_addr}" --master_port="${port}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$PREV_STAGE_CHECKPOINT" \
    --version "$PROMPT_VERSION" \
    --data_path "/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/seg64_fixed_parsed_data/seg64_merged_miradata_84k_train_dataset.csv" \
    --image_folder /home/kaipoc/personal/research_vh/NULL \
    --video_folder /home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/video/clip_video \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model,contrastive_projector,act_squeezer" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "${run_name}" \
    --output_dir "${output_dir}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 22768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 64 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --dehallu_finetune "${dehallu_finetune}" \
    --contrastive_projector_lr "${cp_lr}" \
    --contrastive_projector_weight_decay 0.05 \
    --vccl_wt "${vccl_wt}" \
    --tpocl_wt "${tpocl_wt}" \
    --tpacl_wt "${tpacl_wt}" \
    --use_hard_neg "${use_hard_neg}"

exit 0;