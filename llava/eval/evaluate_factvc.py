import argparse
import os
import sys
from collections import defaultdict
import numpy as np
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from libs.FactVC.facvc_scorer import get_caption, get_factvc_score

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute correlation between FactVC and factuality annotation')
    parser.add_argument('--pred_data_dir', type=str, required=True)
    parser.add_argument('--score_data_dir', type=str, required=True)
    parser.add_argument('--score_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="playground/FactVC/data")
    parser.add_argument('--dataset', type=str, default='activitynet', choices=['activitynet', 'youcook2', 'all'])
    args = parser.parse_args()

    # rename args for lazy modification
    args.pred_file = os.path.join(args.pred_data_dir, args.pred_file)
    
    inference_type = args.dataset
    if inference_type == 'all':
        dataset_list = ['activitynet', 'youcook2']
    else:
        dataset_list = [args.dataset]
    
    overall = {'EMScore(X,X*)': {'full_P': [], 'full_R': [], 'full_F': []}, 
               'EMScore(X,V)': {'full_P': [], 'full_R': [], 'full_F': []},
               'EMScore(X,V,X*)': {'full_P': [], 'full_R': [], 'full_F': []}}
    result = {}
    
    for dataset in dataset_list:
        args.dataset = dataset
        gt_cap, model_cap = get_caption(args)
        factvc_score = get_factvc_score(args, gt_cap, model_cap)
        tmp_result = {'EMScore(X,X*)': {'full_P': 0, 'full_R': 0, 'full_F': 0}, 
                     'EMScore(X,V)': {'full_P': 0, 'full_R': 0, 'full_F': 0},
                     'EMScore(X,V,X*)': {'full_P': 0, 'full_R': 0, 'full_F': 0}}
        
        for key in factvc_score.keys():
            for metric in ("full_P", "full_R", "full_F"):
                tmp = (factvc_score[key][metric]).detach().cpu().numpy()
                overall[key][metric].extend(tmp)
                tmp_result[key][metric] = float(np.mean(tmp))
        result[dataset] = tmp_result
        
        if inference_type == "all":   
            print(f"dataset: {dataset}")
            for key in tmp_result.keys():
                for metric in tmp_result[key].keys():
                    print(f"{key} {metric}: {tmp_result[key][metric]}")
    
    print(f"dataset: {inference_type}")
    for key in overall.keys():
        for metric in overall[key].keys():
            print(f"{key} {metric}: {np.mean(overall[key][metric])}")
            overall[key][metric] = float(np.mean(overall[key][metric]))
    if inference_type == "all":
        result["all"] = overall
    
    os.makedirs(args.score_data_dir, exist_ok=True)
    score_filepath = os.path.join(args.score_data_dir, args.score_file)
    with open(score_filepath, "w") as f:
        json.dump(result, f, indent=4)

