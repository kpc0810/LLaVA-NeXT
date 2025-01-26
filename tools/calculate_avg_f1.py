import os
import glob
import json
import numpy as np
from pprint import pprint

def main(exp_name):
    score_dir = f"outputs/miradata/scores/{exp_name}"
    score_files = glob.glob(os.path.join(score_dir, "score_*.json"))
    score_files = [score_file for score_file in score_files if "parsed_result" not in score_file]
    score_files = sorted(score_files, key=lambda x: int(x.split('_')[-1].split('.')[0].strip('[]')))
    for score_file in score_files:
        with open(score_file, "r") as f:
            score_data = json.load(f)
        print("="*60)
        print(f"Processing {score_file}...")
        all_weighted_f1_act = []
        all_weighted_f1_obj = []
        for round_name, round_data in score_data.items():
            weighted_f1_act = round_data["weighted_f1_act"]
            weighted_f1_obj = round_data["weighted_f1_obj"]
            all_weighted_f1_act.append(weighted_f1_act)
            all_weighted_f1_obj.append(weighted_f1_obj)
            
        print("all_weighted_f1_act: ", all_weighted_f1_act)
        print("all_weighted_f1_obj: ", all_weighted_f1_obj)
        print(f"avg weighted_f1_act: {np.mean(all_weighted_f1_act)}")
        print(f"avg weighted_f1_obj: {np.mean(all_weighted_f1_obj)}")

if __name__ == "__main__":
    exp_name = "llava-qwen-7b_fcl_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"
    main(exp_name)
    print(f"Done!")