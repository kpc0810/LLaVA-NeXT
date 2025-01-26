# !/bin/bash

# params required to change
exp_name="llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"
ckpt_iter=640

pred_data_dir="outputs/factvc/pred_results/${exp_name}"
score_data_dir="outputs/factvc/scores/${exp_name}"
pred_file="pred_[${exp_name}]_[${ckpt_iter}].jsonl"
score_file="score_[${exp_name}]_[${ckpt_iter}].json"
data_dir="/home/kaipoc/personal/research_vh/LLaVA-NeXT/playground/FactVC/data"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=51466 \
    --master_addr "127.0.0.1" --node_rank="0"  \
    llava/eval/evaluate_factvc.py \
    --pred_data_dir "${pred_data_dir}" \
    --score_data_dir "${score_data_dir}" \
    --pred_file "${pred_file}" \
    --score_file "${score_file}" \
    --data_dir "${data_dir}" \
    --dataset "all"