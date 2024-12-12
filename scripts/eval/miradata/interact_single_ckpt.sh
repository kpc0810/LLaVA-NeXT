# !/bin/bash

# params required to change
exp_name="llava-qwen-7b_gl_nnode=8_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1"
ckpt_iter=800

pred_data_dir="outputs/miradata/pred_results/${exp_name}"
score_data_dir="outputs/miradata/scores/${exp_name}"
pred_file="pred_[${exp_name}]_[${ckpt_iter}].json"
score_file="score_[${exp_name}]_[${ckpt_iter}].json"
test_dataset_path="/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/final_miradata_9k_test_dataset.csv"

python llava/eval/evaluate_cap_hallunle_miradata.py \
    --pred_data_dir "${pred_data_dir}" \
    --score_data_dir "${score_data_dir}" \
    --test_dataset_path "${test_dataset_path}" \
    --pred_file "${pred_file}" \
    --score_file "${score_file}"