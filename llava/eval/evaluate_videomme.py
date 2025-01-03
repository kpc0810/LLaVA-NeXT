import os
import json
import numpy as np
import prettytable
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="outputs/videomme/videomme_nosub.jsonl")
    parser.add_argument("--score_file", type=str, default="outputs/videomme/videomme_nosub_score.json")
    parser.add_argument("--gt_file", type=str, default="playground/videomme/qa_old_format.json")
    return parser.parse_args()

def main(args):
    
    result = defaultdict(list)
    
    pred_data = {}
    with open(args.pred_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            pred_data[sample["question_id"]] = sample["prediction"]
    gt_data = {}
    for sample in json.load(open(args.gt_file, "r")):
        for q in sample["questions"]:
            gt_data[q["question_id"]] = {
                "duration_category": sample["duration_category"],
                "answer": q["answer"]
            }
    
    for qid, sample in gt_data.items():
        pred = pred_data.get(qid, "")
        pred = pred.replace(".", "").strip().upper()[0] if pred else ""
        pred = int(pred == sample["answer"])
        result[sample["duration_category"]].append(pred)
        
    
    table = prettytable.PrettyTable()
    table.field_names = ["Category", "Accuracy", "Sample Count"]
    
    table.align["Category"] = "l"
    table.align["Accuracy"] = "c" 
    table.align["Sample Count"] = "c"
    overall = []
    out = {}
    for category, preds in result.items():
        acc = np.mean(preds) * 100
        table.add_row([category.capitalize(), f"{acc:.2f}%", len(preds)])
        overall.extend(preds)
        out[category] = acc
    table.add_row(["Overall", f"{np.mean(overall) * 100:.2f}%", len(overall)])
    out["overall"] = np.mean(overall) * 100
    print(table)
    
    # Save the table to a file
    try:
        save_dir = os.path.dirname(args.score_file)
        os.makedirs(save_dir, exist_ok=True)
        score_file = args.score_file
    except :
        score_file = args.pred_file.replace(".jsonl", "_score.json")
    with open(score_file, 'w') as f:
        json.dump(out, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)