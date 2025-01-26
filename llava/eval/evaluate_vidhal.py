import os
import json
import numpy as np
import prettytable
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_data_dir", type=str, required=True)
    parser.add_argument("--score_data_dir", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--score_file", type=str, required=True)
    parser.add_argument("--gt_file", type=str, required=True)
    return parser.parse_args()

def main(args):
    
    result = defaultdict(list)
    fail_case = defaultdict(int)
    args.pred_file = os.path.join(args.pred_data_dir, args.pred_file)
    
    pred_data = {}
    with open(args.pred_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            pred_data[sample["video"]] = {
                "prediction": sample["prediction"],
                "answer": sample["answer"]
            }
    gt_data = json.load(open(args.gt_file, "r"))
    
    for sample in gt_data:
        vid = sample["video"]
        pred = pred_data[vid]["prediction"]
        answer = pred_data[vid]["answer"]
        pred = pred.replace(".", "").strip().upper()[0] if pred else ""
        if pred == "":
            fail_case[sample["aspect"]] += 1
        pred = int(pred == answer)
        result[sample["aspect"]].append(pred)
    
    table = prettytable.PrettyTable()
    table.field_names = ["Category", "Accuracy", "Sample Count", "Fail Case"]
    
    table.align["Category"] = "l"
    table.align["Accuracy"] = "c" 
    table.align["Sample Count"] = "c"
    table.align["Fail Case"] = "c"
    overall = []
    out = {}
    for category, preds in result.items():
        acc = np.mean(preds) * 100
        table.add_row([category.capitalize(), f"{acc:.2f}%", len(preds), fail_case[category]])
        overall.extend(preds)
        out[category] = {
            "accuracy": acc,
            "sample_count": len(preds),
            "fail_case": fail_case[category]
        }
    table.add_row(["Overall", f"{np.mean(overall) * 100:.2f}%", len(overall), sum(fail_case.values())])
    out["overall"] = {
        "accuracy": np.mean(overall) * 100,
        "sample_count": len(overall),
        "fail_case": sum(fail_case.values())
    }
    print(table)
    
    # Save the table to a file
    os.makedirs(args.score_data_dir, exist_ok=True)
    score_file = os.path.join(args.score_data_dir, args.score_file)
    with open(score_file, 'w') as f:
        json.dump(out, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)