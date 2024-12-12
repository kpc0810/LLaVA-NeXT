import os
import argparse
import json
import csv
import ast
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
from collections import defaultdict

from llava.model.utils import parse_object_nouns_and_action_verbs
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_compute

import spacy
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_trf")

def parse_args():
    parser = argparse.ArgumentParser(description="calculate nle score for miradata")
    parser.add_argument("--pred_data_dir", required=True, help="The path to get/generate prediction file and score file.")
    parser.add_argument("--score_data_dir", required=True, help="The path to get/generate prediction file and score file.")
    parser.add_argument("--test_dataset_path", required=True, help="The path to get parsed object and action words")
    parser.add_argument("--pred_file", required=True, help="The path to file containing prediction.")
    parser.add_argument("--score_file", required=True, help="The path to file containing data.")
    args = parser.parse_args()
    return args

def compute_sentence_similarity(pds, gts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cos_sims = []
    for pd, gt in zip(pds, gts):
        pd_embedding = model.encode(pd, convert_to_tensor=True)
        gt_embedding = model.encode(gt, convert_to_tensor=True)
        cos_sim = util.cos_sim(pd_embedding, gt_embedding)
        cos_sims.append(cos_sim.item())
    return np.mean(cos_sims)

def calculate_coverage(clip_id, pd, test_data_dict):
    """
    Calculate the coverage of predicted objects and actions against the gpt4o parsed relation pairs.
    
    Args:
        clip_id: The identifier for the video clip.
        pd: The predicted caption.
        test_data_dict: The dictionary containing gpt4o parsed relation pairs for each clip.
        
    Returns:
        tuple: A tuple containing two floats:
            - coverage_obj (float): The coverage of predicted objects against parsed relation pairs.
            - coverage_act (float): The coverage of predicted actions against parsed relation pairs.
    """
    gt_obj_words = []
    gt_act_words = []
    cap_relation_pair = eval(test_data_dict[clip_id])
    for pair in cap_relation_pair:
        gt_act_words += list(pair[:1])
        gt_obj_words += list(pair[1:])
    
    len_gt_obj_words, len_gt_act_words = len(gt_obj_words), len(gt_act_words)
    
    coveage_obj_num = 0
    for gt_obj in gt_obj_words:
        if gt_obj in pd:
            coveage_obj_num += 1
            gt_obj_words.remove(gt_obj)
    
    coveage_act_num = 0
    for gt_act in gt_act_words:
        if gt_act in pd:
            coveage_act_num += 1
            gt_act_words.remove(gt_act)
    
    return coveage_obj_num / len_gt_obj_words, coveage_act_num / len_gt_act_words


def calculate_chair(pd, gt):
    """
    Use spacy to parse the prediction and ground truth, and calculate the chair score of the prediction.
    """
    
    pd_object_nouns, pd_action_verbs = parse_object_nouns_and_action_verbs(pd)
    num_pd_object_nouns, num_pd_action_verbs = len(pd_object_nouns), len(pd_action_verbs)
    
    gt_copy = gt
    num_intersect_object_nouns = 0
    for obj in pd_object_nouns:
        if obj in gt_copy:
            gt_copy = gt_copy.replace(obj, "")
            num_intersect_object_nouns += 1
        else:
            stemmed_obj = stemmer.stem(obj)
            if stemmed_obj in gt_copy:
                gt_copy = gt_copy.replace(stemmed_obj, "")
                num_intersect_object_nouns += 1
    
    gt_copy = gt
    num_intersect_verb_verbs = 0
    for act in pd_action_verbs:
        if act in gt_copy:
            gt_copy = gt_copy.replace(act, "")
            num_intersect_verb_verbs += 1
        else:
            stemmed_act = stemmer.stem(act)
            if stemmed_act in gt_copy:
                gt_copy = gt_copy.replace(stemmed_act, "")
                num_intersect_verb_verbs += 1
    
    chair_obj = 1 - num_intersect_object_nouns / num_pd_object_nouns if num_pd_object_nouns > 0 else 1
    chair_act = 1 - num_intersect_verb_verbs / num_pd_action_verbs if num_pd_action_verbs > 0 else 1
    
    return chair_obj, chair_act


def compute_hallucination_score(clip_ids, pds, gts, round,test_dataset_path):
    hallu_eval_dict = {
        'chair_obj': [], 
        'chair_act': [], 
        'coverage_obj': [], 
        'coverage_act': [], 
        'hallu_obj': [], 
        'hallu_act': []
    }
    
    # load field {$round}_relation_pair 
    test_data_dict = {}
    with open(test_dataset_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            clip_id = row['clip_id']
            relation_pair = row[f'{round}_relation_pair']
            test_data_dict[clip_id] = relation_pair
    
    for clip_id, pd, gt in zip(clip_ids, pds, gts):
        
        if clip_id not in test_data_dict:
            print("Skip clip_id:", clip_id)
            continue
        
        pd, gt = pd.lower(), gt.lower()
        
        # CHAIR
        chair_obj, chair_act = calculate_chair(pd, gt)
        hallu_eval_dict["chair_obj"].append(chair_obj)
        hallu_eval_dict["chair_act"].append(chair_act)
        
        # Coverage
        coverage_obj, coverage_act = calculate_coverage(clip_id, pd, test_data_dict)
        hallu_eval_dict["coverage_obj"].append(coverage_obj)
        hallu_eval_dict["coverage_act"].append(coverage_act)
        
        # Hal
        hallu_eval_dict["hallu_obj"].append(1 if chair_obj != 0 else 0)
        hallu_eval_dict["hallu_act"].append(1 if chair_act != 0 else 0)
    
    for k, v in hallu_eval_dict.items():
        print("%s: %.2f%%"%(k, np.mean(v)*100))
    
    hallu_eval_dict = {k: np.mean(v) for k, v in hallu_eval_dict.items()}
    for k, v in hallu_eval_dict.items():
        print(f"{k}: {v:.2f}")
    
    return hallu_eval_dict

def calculate_score(samples, round, test_dataset_path):
    # Dictionary to store the count of occurrences for each video_id
    pds = defaultdict(list)
    gts = defaultdict(list)
    pds_all = []
    gts_all = []
    clip_id_all = []
    # Iterate through each sample in pred_contents
    for i, sample in enumerate(samples):
        sample = json.loads(sample)
        clip_id = sample['clip_id']
        pred_caption = sample['pred_captions'][round]
        gt_caption = sample['gt_captions'][round]
        pds[i].append({'image_id': clip_id, 'caption': pred_caption})
        pds_all += [pred_caption]
        gts[i].append({'image_id': clip_id, 'caption': gt_caption})
        gts_all += [gt_caption]
        clip_id_all += [clip_id]
        
    tokenizer = PTBTokenizer()
    tok_pds = tokenizer.tokenize(pds)
    tok_gts = tokenizer.tokenize(gts)
    
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
               (Meteor(),"METEOR"),
               (Rouge(), "ROUGE_L"),
               (Cider(), "CIDEr")]

    eval_dict = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tok_gts, tok_pds)
        if scorer.method() == "Bleu":
            eval_dict["BLEU4"] = score[3]
        else:
            eval_dict[scorer.method()] = score

    _, _, score = bert_score_compute(pds_all, gts_all, lang='en', verbose=False)
    eval_dict["BERTScore"] = score.mean().item()
    
    eval_dict["Semantic Textual Similarity"] = compute_sentence_similarity(pds_all, gts_all)
    hallu_eval_dict = compute_hallucination_score(clip_id_all, pds_all, gts_all, round, test_dataset_path)
    eval_dict.update(hallu_eval_dict)

    
    for k, v in eval_dict.items():
        print("%s: %.2f%%"%(k, v*100))
    
    return eval_dict

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    
    pred_path = os.path.join(args.pred_data_dir, f"{args.pred_file}")
    samples = open(pred_path).readlines()
    
    score_path = os.path.join(args.score_data_dir, f"{args.score_file}")
    os.makedirs(args.score_data_dir, exist_ok=True)
    all_scores = {}
    
    rounds = ['dense', 'main_object', 'background']
    for round in tqdm(rounds):
        eval_dict = calculate_score(samples, round, args.test_dataset_path)
        all_scores[f"{round}"] = eval_dict
    
    print(f"Save scores to {score_path}")
    with open(score_path, 'w') as f:
        json.dump(all_scores, f, indent=4)
    


if __name__ == "__main__":
    main()
    print("Done!")
