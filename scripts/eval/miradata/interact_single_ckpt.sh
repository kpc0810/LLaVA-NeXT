# !/bin/bash

# params required to change
exp_name="llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"
ckpt_iter=1000

pred_data_dir="outputs/miradata/pred_results/${exp_name}"
score_data_dir="outputs/miradata/scores/${exp_name}"
pred_file="pred_[${exp_name}]_[${ckpt_iter}].json"
score_file="score_[${exp_name}]_[${ckpt_iter}].json"
test_dataset_path="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/final_miradata_9k_test_dataset.csv"
debug=True

torchrun --nnodes=1 --nproc_per_node=1 --master_port=51466 \
    --master_addr "127.0.0.1" --node_rank="0"  \
    llava/eval/evaluate_cap_hallu_miradata.py \
    --pred_data_dir "${pred_data_dir}" \
    --score_data_dir "${score_data_dir}" \
    --test_dataset_path "${test_dataset_path}" \
    --pred_file "${pred_file}" \
    --score_file "${score_file}" \
    --debug "${debug}"