#!/bin/bash

# Define the experiments array
experiments=(
    "llava-qwen-7b_gl_nnode=8_epo=1_plmlr=1e-4_vtlr=1e-6_bspd=1"
    "llava-qwen-7b_gl_nnode=8_epo=1_plmlr=5e-5_vtlr=1e-6_bspd=1"
    "llava-qwen-7b_gl_nnode=8_epo=1_plmlr=5e-6_vtlr=1e-6_bspd=1"
    "llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"
)
cd /home/kaipoc/personal/research_vh/LLaVA-NeXT

# Loop through each experiment
chmod +x scripts/eval/miradata/lunch_ckpts.sh
for exp_name in "${experiments[@]}"; do
    echo "=== Running experiment: $exp_name ==="
    bash scripts/eval/miradata/lunch_ckpts.sh "$exp_name" 3
    echo "=== Finished experiment: $exp_name ==="
done