# Standard library imports
import argparse
import asyncio
import ast
import base64
import copy
import csv
import json
import math
import os
import random
import signal
import time
import glob
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
import fcntl

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.distributed as dist
import transformers
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer, util

# Local application/library specific imports
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.datamodule.mira_user_prompt import user_prompts
from llava.model.builder import load_pretrained_faith_model
from llava.utils import process_video_with_decord, disable_torch_init, rank_print
from llava.mm_utils import (
    process_anyres_image,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from llava.model.language_model.faith_llava_qwen import all_gather

import spacy
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_lg")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def split_list_by_len(lst, n):
    # Initialize empty chunks and the total length for each chunk
    chunks = [[] for _ in range(n)]
    chunk_lengths = [0] * n  # Track total length for each chunk
    
    # Sort the elements by length in descending order for a greedy allocation
    sorted_lst = sorted(lst, key=len, reverse=True)
    
    # Distribute elements greedily to the chunk with the least total length
    for item in sorted_lst:
        # Find the chunk with the smallest current length
        min_index = chunk_lengths.index(min(chunk_lengths))
        chunks[min_index].append(item)
        chunk_lengths[min_index] += len(item)
    
    return chunks

def get_chunk(lst, n, k):
    chunks = split_list_by_len(lst, n)
    return chunks[k]


def parse_args():
    parser = argparse.ArgumentParser(description="calculate nle score for miradata")
    parser.add_argument("--pred_data_dir", required=True, help="The path to get/generate prediction file and score file.")
    parser.add_argument("--score_data_dir", required=True, help="The path to get/generate prediction file and score file.")
    parser.add_argument("--test_dataset_path", required=True, help="The path to get parsed object and action words")
    parser.add_argument("--pred_file", required=True, help="The path to file containing prediction.")
    parser.add_argument("--score_file", required=True, help="The path to file containing data.")
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    return args


def process_one_sample(pd, model):
        prompt = f"Now, perform dependency parsing on the given detailed caption according to the steps from (1) to (3).\nInput: {pd}\n"
        messages = [
            {"role": "system", "content": (
                "You are an expert in dependency parsing. Your task is to process a detailed caption describing a video and output the results in the form of dependency parsing tuples. "
                "Note that the string in each tuple must be a single word or a phrase that is exactly present in the original caption. "
                "Follow the steps below carefully:"
                "(1) Identify all objects mentioned in the caption that are both physically visible and clearly recognizable in the video."
                "**[IMPORTANT]: Exclude any objects that are too abstract or cannot be visually recognized in the video (e.g., concepts like 'freedom' or 'thoughts')** "
                "(2) Identify all action verbs in the caption that are performed by the objects recognized in step (1). "
                "**[IMPORTANT]: Exclude any verbs that describe abstract actions, mental states, emotions, or general characteristics (e.g., 'highlight', 'believe', 'know', 'exist', 'indicate', 'attempt', 'try', 'engage' 'face' or 'suggest' must be excluded) as well as be verbs (e.g., 'is', 'are', 'was', 'were' must be excluded).**"
                "**Only include verbs describing actions that can be observed visually in the video.**"
                "(3) For each action verb identified in step (2), find the corresponding object(s) recognized in step (1) that are performing or involved in the action. "
                "If the action verb is linked to a single object, output a tuple in the format: **(VERB, OBJECT)**. "
                "If the action verb involves two objects (e.g., a subject and an object), output a tuple in the format: **(VERB, OBJECT1, OBJECT2)**. "
                "Note that every action verb and object in the tuples must be present in the list of objects and action verbs recognized in step (1) and (2)."
                "Finally, combine all tuples into a single paragraph, ensuring the list of tuples is presented as continuous text without line breaks, bullet points, or any additional formatting.\n"
                "<EXAMPLES>\n"
                "Input: "
                "A group of female basketball players practicing in a gym. The players are divided into two teams, wearing yellow and purple jerseys. "
                "They are actively engaged in passing the ball, dribbling, and attempting shots. "
                "The gym is well-lit with natural light streaming through large windows, and the wooden floor is marked with basketball court lines. "
                "The players exhibit teamwork and coordination as they move dynamically across the court.\n"
                "Output: "
                "(1) List of Object(s): ['basketball players', 'gym', 'jerseys', 'ball', 'shots', 'windows', 'floor', 'basketball court lines']"
                "(2) List of Action Verb(s): ['practicing', 'wearing', 'passing', 'dribbling', 'marked']"
                "(3) List of Dependency Parsing Tuple(s): [('practicing', 'basketball players'), ('wearing', 'basketball players', 'jerseys'), ('passing', 'basketball players', 'balls'), ('dribbling', 'basketball players', 'balls'), ('marked', 'floor', 'basketball court lines')]\n"
                "Input: "
                "A first-person perspective of a motorcycle ride through a bustling cityscape, likely on a rainy day given the wet streets and overcast sky. "
                "The rider maneuvers through various urban settings, including commercial areas filled with shops and billboards, and quieter residential zones. "
                "The city is vibrant, with neon signs and advertisements adding color and life to the grey, damp environment. "
                "The journey gives a dynamic view of city life from the unique vantage point of a motorcyclist, highlighting both the broad avenues and the more intimate alleyways of the urban landscape.\n"
                "Output: "
                "(1) List of Object(s): ['motorcycle', 'rider', 'streets', 'sky', 'shops', 'billboards', 'neon signs', 'advertisements', 'motorcyclist', 'avenues', 'alleyways']"
                "(2) List of Action Verb(s): ['ride', 'maneuvers']"
                "(3) List of Dependency Parsing Tuple(s): [('ride', 'motorcycle'), ('maneuvers', 'rider', 'shops'), ('maneuvers', 'rider', 'billboards')]\n"
                "Input: "
                "A virtual motorcycle ride through a bustling urban environment in a video game. "
                "The rider, clad in a yellow jacket, navigates the motorcycle through various cityscapes, "
                "including back alleys, main streets, and highways. The city is rich in detail, with dynamic "
                "weather conditions and a diverse array of pedestrians and vehicles populating the streets. "
                "The journey captures the essence of a high-speed chase or a time-sensitive mission, with the "
                "rider skillfully maneuvering around obstacles and traffic.\n"
                "Output: "
                "(1) List of Object(s): ['motorcycle', 'rider', 'jacket', 'alleys', 'streets', 'highways', 'pedestrians', 'vehicles', 'streets', 'obstacles', 'traffic']"
                "(2) List of Action Verb(s): ['ride', 'clad', 'navigates', 'populating', 'maneuvering']"
                "(3) List of Dependency Parsing Tuple(s): [('ride', 'motorcycle'), ('clad', 'rider', 'jacket'), ('navigate', 'rider', 'motorcycle'), ('populating', 'vehicles', 'streets'), ('populating', 'pedestrians', 'streets'), ('maneuver', 'rider', 'obstacles), ('maneuver', 'rider', 'traffic')]\n"
                "Input: " 
                "Features towering skyscrapers adorned with bright advertisements and corporate logos, "
                "suggesting a commercial district in a highly developed urban setting. "
                "The architecture displays a mix of high-tech design with sleek, metallic surfaces and occasional green spaces "
                "that add a touch of nature to the metallic urban environment. "
                "The consistent rainfall and reflective wet surfaces enhance the night-time setting, emphasizing the city's vibrant nightlife and continuous activity.\n"
                "Output: "
                "(1) List of Object(s): ['skyscrapers', 'advertisements', 'logos', 'pedestrians', 'vehicles', 'streets', 'obstacles', 'traffic']"
                "(2) List of Action Verb(s): ['adorned']"
                "(3) List of Dependency Parsing Tuple(s): [('adorned', 'skyscrapers', 'advertisements'), ('adorned', 'skyscrapers', 'logos')]\n"
                "<END_OF_EXAMPLES>\n"
                )
            },
            {"role": "user", "content": prompt}
        ]
        outputs = model(
            messages,
            max_new_tokens=500,
        )
        response = outputs[0]["generated_text"][-1]["content"]  
        return response


def postprocess(text):
    # Adjusted start pattern to match "Tuple:", "Tuples:", or "Tuple(s):"
    start_pattern = r"\(3\) List of Dependency Parsing Tuple(?:\(s\)|s)?:"
    
    # Find the start of the tuple list
    start_match = re.search(start_pattern, text)
    if not start_match:
        return None

    # Extract text after the starting pattern
    text_after = text[start_match.end():]

    # Find the end of the tuple list by looking for two consecutive newlines, the start of the next section, or the end of the text
    end_pattern = r"\n\s*\n|\n\(\d+\)|\Z"
    end_match = re.search(end_pattern, text_after)
    if end_match:
        tuples_text = text_after[:end_match.start()]
    else:
        tuples_text = text_after

    # Define a regex pattern to match tuples with or without quotes
    tuple_pattern = r"\(\s*(?:'[^']*'|\"[^\"]*\"|[^,()]+)\s*(?:,\s*(?:'[^']*'|\"[^\"]*\"|[^,()]+)\s*){0,2}\)"

    # Find all matches of tuples in the tuples_text
    tuples = re.findall(tuple_pattern, tuples_text)

    # Parse each tuple string
    parsed_tuples = []
    for t_str in tuples:
        # Clean up the tuple string
        t_str_clean = t_str.strip("() ")
        # Split the tuple string by commas
        elements = [elem.strip(" '\"") for elem in t_str_clean.split(",")]
        # Only include tuples with at least two elements
        if len(elements) >= 2:
            t = tuple(elements)
            if t not in parsed_tuples:
                parsed_tuples.append(t)
    return parsed_tuples if parsed_tuples else []


def parse_object_nouns_and_action_verbs_by_llm(model, args, pd, local_rank):
    max_retry, cur_retry = 5, 0
    while cur_retry < max_retry:
        print("Try to parse object and action verbs by llm in %d times"%cur_retry)
        response = process_one_sample(pd, model)
        relation_pair = postprocess(response)
        if relation_pair is not None:
            break
        cur_retry += 1
    if cur_retry == max_retry:
        return None, None

    object_nouns, action_verbs = [], []

    for pair in relation_pair:
        action_verbs += list(pair[:1])
        object_nouns += list(pair[1:])

    return object_nouns, action_verbs

class HalluScorer:
    def __init__(self, data):
        self.data = data
        self.tfidf_obj, self.tfidf_act = self._build_tfidf_index()

    def _build_tfidf_index(self):
        """
        Calculate TF-IDF without simplification:
          1. First collect the occurrence count (term frequency, TF) of object words and action words for each sample (document).
          2. Calculate document frequency (DF) for all words to obtain IDF.
          3. Calculate TF-IDF for each (document, word) pair.
        
        Returns:
          doc2tfidf_obj, doc2tfidf_act
            => dict[doc_id][word] = TF-IDF score
        """

        doc2wordcount_obj = {}
        doc2wordcount_act = {}
        df_obj = {}
        df_act = {}

        for sample in self.data:
            doc_id = sample['clip_id']
            doc2wordcount_obj[doc_id] = {}
            doc2wordcount_act[doc_id] = {}

            # dense_relation_pair, main_object_relation_pair, and background_relation_pair are considered as one doc
            for round in ['dense', 'main_object', 'background']:
                round_rel_pairs = sample[f"{round}_relation_pair"]
                doc_act_words = []
                doc_obj_words = []
                for pair in round_rel_pairs:
                    doc_act_words += list(pair[:1])
                    doc_obj_words += list(pair[1:])

                for w in doc_obj_words:
                    doc2wordcount_obj[doc_id][w] = doc2wordcount_obj[doc_id].get(w, 0) + 1
                for w in doc_act_words:
                    doc2wordcount_act[doc_id][w] = doc2wordcount_act[doc_id].get(w, 0) + 1

            for w in doc2wordcount_obj[doc_id]:
                df_obj[w] = df_obj.get(w, 0) + 1
            for w in doc2wordcount_act[doc_id]:
                df_act[w] = df_act.get(w, 0) + 1

        # IDF = log( (N+1)/(df(w)+1 ) ) + 1
        N = len(self.data)  # N is the number of documents
        idf_obj = {}
        idf_act = {}
        for w, doc_freq in df_obj.items():
            idf_obj[w] = math.log((N + 1)/(doc_freq + 1)) + 1
        for w, doc_freq in df_act.items():
            idf_act[w] = math.log((N + 1)/(doc_freq + 1)) + 1

        # calculate tf-idf for each (doc, word)
        doc2tfidf_obj = {}
        doc2tfidf_act = {}
        for data in self.data:
            doc_id = data['clip_id']
            doc2tfidf_obj[doc_id] = {}
            doc2tfidf_act[doc_id] = {}
            
            sum_obj_counts = sum(doc2wordcount_obj[doc_id].values())
            for w, count in doc2wordcount_obj[doc_id].items():
                tf = count / sum_obj_counts if sum_obj_counts > 0 else 0.0  # normalize tf
                tfidf = tf * idf_obj.get(w, 0.0)
                doc2tfidf_obj[doc_id][w] = tfidf
            sum_act_counts = sum(doc2wordcount_act[doc_id].values())
            for w, count in doc2wordcount_act[doc_id].items():
                tf = count / sum_act_counts if sum_act_counts > 0 else 0.0
                tfidf = tf * idf_act.get(w, 0.0)
                doc2tfidf_act[doc_id][w] = tfidf

        return doc2tfidf_obj, doc2tfidf_act

    def process_one_sample(self, sample, round):
        self.__reset()
        self.sample = sample
        self.clip_id = sample['clip_id']
        self.pred_obj_nouns = sample['parsed_keywords_from_pred_captions'][round]['object_nouns']
        self.pred_act_verbs = sample['parsed_keywords_from_pred_captions'][round]['action_verbs']
        self.gt_cap = sample['gt_captions'][round]
        self.pd_cap = sample['pred_captions'][round]
        self.gt_relation_pair = sample[f'{round}_relation_pair']
        
        # Extract ground truth words once during initialization
        self.gt_act_words, self.gt_obj_words = [], []
        for gt_pair in self.gt_relation_pair:
            self.gt_act_words += list(gt_pair[:1])
            self.gt_obj_words += list(gt_pair[1:])

    def __reset(self):
        self.sample = None
        self.pred_obj_nouns = None
        self.pred_act_verbs = None
        self.gt_cap = None
        self.pd_cap = None
        self.gt_relation_pair = None
        self.gt_act_words = []
        self.gt_obj_words = []
        
    def _count_intersections(self, pred_words, gt_text):
        """Helper method to count intersections between predicted words and ground truth text"""
        intersect_count = 0
        gt_copy = gt_text
        for word in pred_words:
            if word in gt_copy:
                gt_copy = gt_copy.replace(word, "")
                intersect_count += 1
            else:
                stemmed_word = stemmer.stem(word)
                if stemmed_word in gt_copy:
                    gt_copy = gt_copy.replace(stemmed_word, "")
                    intersect_count += 1
        return intersect_count


    def _get_max_spacy_similarity(self, pred_word, gt_words):
        """
        Calculate the maximum similarity between a predicted word and a list of ground-truth words using SpaCy vectors.
        similarity = cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
        """
        doc_word = nlp(pred_word)
        # If the token has no vector (vector_norm=0), return 0.0
        if doc_word.vector_norm == 0:
            return 0.0
        
        word_vec = doc_word.vector
        word_vec_norm = doc_word.vector_norm

        max_sim = 0.0
        for gt_word in gt_words:
            doc_gt_word = nlp(gt_word)
            if doc_gt_word.vector_norm == 0:
                continue
            sim = (word_vec @ doc_gt_word.vector) / (word_vec_norm * doc_gt_word.vector_norm)
            if sim > max_sim:
                max_sim = sim
        
        return max_sim

    def one_sample_calculate_chair(self):
        num_pd_obj = len(self.pred_obj_nouns)
        num_pd_act = len(self.pred_act_verbs)
        
        obj_intersect = self._count_intersections(self.pred_obj_nouns, self.gt_cap)
        act_intersect = self._count_intersections(self.pred_act_verbs, self.gt_cap)
        
        chair_obj = 1 - obj_intersect / num_pd_obj if num_pd_obj > 0 else 1
        chair_act = 1 - act_intersect / num_pd_act if num_pd_act > 0 else 1
        
        return chair_obj, chair_act

    def one_sample_calculate_coverage(self):
        len_gt_obj = len(self.gt_obj_words)
        len_gt_act = len(self.gt_act_words)
        
        # Create copies to avoid modifying original lists
        gt_obj_copy = self.gt_obj_words.copy()
        gt_act_copy = self.gt_act_words.copy()
        
        cover_obj = sum(1 for word in gt_obj_copy if word in self.pd_cap)
        cover_act = sum(1 for word in gt_act_copy if word in self.pd_cap)
        
        return cover_obj / len_gt_obj, cover_act / len_gt_act
    
    def one_sample_calculate_weighted_chair(self):
        num_pd_obj = len(self.pred_obj_nouns)
        num_pd_act = len(self.pred_act_verbs)
        
        if num_pd_obj == 0 or num_pd_act == 0:
            return 1.0, 1.0
        
        obj_total_sim = 0.0
        for pd_obj in self.pred_obj_nouns:
            max_sim_obj = self._get_max_spacy_similarity(pd_obj.lower(), self.gt_obj_words)
            obj_total_sim += max_sim_obj
        avg_obj_sim = obj_total_sim / num_pd_obj if num_pd_obj > 0 else 0.0

        act_total_sim = 0.0
        for pd_act in self.pred_act_verbs:
            max_sim_act = self._get_max_spacy_similarity(pd_act.lower(), self.gt_act_words)
            act_total_sim += max_sim_act
        avg_act_sim = act_total_sim / num_pd_act if num_pd_act > 0 else 0.0

        return 1-avg_obj_sim, 1-avg_act_sim
    
    def one_sample_calculate_weighted_coverage(self):
        """
        Calculate weighted coverage based on the following logic:
        - If a ground-truth word appears in self.pd_cap, add its tf-idf score to the numerator.
        - The denominator is the sum of the tf-idf scores of all ground-truth words.
        - Coverage = numerator / denominator
        """
        # If there are no gt_obj_words and gt_act_words in this sample, return (0, 0)
        if len(self.gt_obj_words) == 0 and len(self.gt_act_words) == 0:
            return 0.0, 0.0

        denom_obj = 0.0
        numer_obj = 0.0

        for gt_obj in self.gt_obj_words:
            tfidf_val = self.tfidf_obj[self.clip_id][gt_obj]
            denom_obj += tfidf_val
            if gt_obj in self.pd_cap:
                numer_obj += tfidf_val
        weighted_coverage_obj = numer_obj / denom_obj if denom_obj > 0 else 0.0

        denom_act = 0.0
        numer_act = 0.0
        for gt_act in self.gt_act_words:
            tfidf_val = self.tfidf_act[self.clip_id][gt_act]
            denom_act += tfidf_val
            if gt_act in self.pd_cap:
                numer_act += tfidf_val
        weighted_coverage_act = numer_act / denom_act if denom_act > 0 else 0.0

        return weighted_coverage_obj, weighted_coverage_act


def run_hallu_eval(args):
    """
    Run evaluation on Miradata Captioning DataSet using the LLaVA-Video model.

    Args:
        args: Command-line arguments.
    """
    # Intialize the distributed environment
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)  # This sets the current GPU device to the one corresponding to the local rank

    ## prepare parsed data
    # load cached parsed data if exists to avoid re-parsing
    os.makedirs(args.score_data_dir, exist_ok=True)
    score_filename, _ = os.path.splitext(os.path.basename(args.score_file))
    parsed_file = f"{score_filename}.parsed_result.json"
    parsed_filepath = os.path.join(args.score_data_dir, parsed_file)
    
    cached_parsed_ids, cached_parsed_data = [], []
    if not os.path.exists(parsed_filepath):
        open(parsed_filepath, "w").close()
    else:
        with open(parsed_filepath, "r") as f:
            cached_parsed_data = f.readlines()
            cached_parsed_ids = [json.loads(line)['clip_id'] for line in cached_parsed_data]

    # load prediction data and skip the cached ids
    pred_filepath = os.path.join(args.pred_data_dir, args.pred_file)
    with open(pred_filepath, "r") as f:
        pred_data = f.readlines()
        pred_data = [json.loads(sample) for sample in pred_data]
    pred_data = [sample for sample in pred_data if sample['clip_id'] not in cached_parsed_ids]
    pred_data = get_chunk(pred_data, world_size, local_rank) 
    
    if args.debug:
        save_every_n_sample = 0
    else:
        save_every_n_sample = 50

    rounds = ['dense', 'main_object', 'background']
    if pred_data != []:
        # continue to parse the prediction data 
        model = transformers.pipeline(
            "text-generation",
            model=args.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=local_rank,
            temperature = 0.3,
        )

        added_parsed_data = []
        for i in tqdm(range(0, len(pred_data))):
            parsed_keywords = {}
            sample = pred_data[i]
            clip_id, pred_captions, gt_captions = sample['clip_id'], sample['pred_captions'], sample['gt_captions']
            
            parser_error = False
            for round in rounds:
                pd = pred_captions[round]
                object_nouns, action_verbs = parse_object_nouns_and_action_verbs_by_llm(model, args, pd, local_rank)
                if object_nouns is None and action_verbs is None:
                    parser_error = True
                    break
                parsed_keywords[round] = {
                    'object_nouns': object_nouns,
                    'action_verbs': action_verbs
                }
            if parser_error:
                continue
            sample['parsed_keywords_from_pred_captions'] = parsed_keywords
            added_parsed_data.append(sample)
            
            # save
            if len(added_parsed_data) > save_every_n_sample:
                with open(parsed_filepath, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    for sample in added_parsed_data:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fcntl.flock(f, fcntl.LOCK_UN)
                added_parsed_data = []
    
    ## calculate hallucination score (parsing is done)
    # load relation pair data
    test_data_dict = {}
    with open(args.test_dataset_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            clip_id = row['clip_id']
            relation_pair = {}
            for round in rounds:
                relation_pair[f'{round}_relation_pair'] = ast.literal_eval(row[f'{round}_relation_pair'])
            test_data_dict[clip_id] = relation_pair
    
    # load parsed data
    with open(parsed_filepath, "r") as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]

    # add relation pair to each sample
    for sample in data:
        for round in rounds:
            sample[f'{round}_relation_pair'] = test_data_dict[sample['clip_id']][f'{round}_relation_pair']
    
    hallu_scorer = HalluScorer(data)
    all_scores = {}
    for round in rounds:
        hallu_eval_dict = {
            'chair_obj': [], 
            'chair_act': [], 
            'coverage_obj': [], 
            'coverage_act': [], 
            'weighted_chair_obj': [], 
            'weighted_chair_act': [],
            'weighted_coverage_obj': [],
            'weighted_coverage_act': []
        }
        for i, sample in enumerate(data):
            print(f"Processing sample {i} of {len(data)}")
            hallu_scorer.process_one_sample(sample, round)
            ## calculate chair hallucination score
            chair_obj, chair_act = hallu_scorer.one_sample_calculate_chair()
            hallu_eval_dict["chair_obj"].append(chair_obj)
            hallu_eval_dict["chair_act"].append(chair_act) 
            
            ## calculate weighted hallucination score
            weighted_chair_obj, weighted_chair_act = hallu_scorer.one_sample_calculate_weighted_chair()
            hallu_eval_dict["weighted_chair_obj"].append(weighted_chair_obj)
            hallu_eval_dict["weighted_chair_act"].append(weighted_chair_act)
            
            ## calculate coverage score
            coverage_obj, coverage_act = hallu_scorer.one_sample_calculate_coverage()
            hallu_eval_dict["coverage_obj"].append(coverage_obj)
            hallu_eval_dict["coverage_act"].append(coverage_act)

            ## calculate weighted_coverage score
            weighted_coverage_obj, weighted_coverage_act = hallu_scorer.one_sample_calculate_weighted_coverage()
            hallu_eval_dict["weighted_coverage_obj"].append(weighted_coverage_obj)
            hallu_eval_dict["weighted_coverage_act"].append(weighted_coverage_act)
        
        all_scores[round] = hallu_eval_dict
    
    score_path = os.path.join(args.score_data_dir, f"{args.score_file}")
    with open(score_path, 'w') as f:
        json.dump(all_scores, f, indent=4)

    return


if __name__ == "__main__":
    args = parse_args()
    run_hallu_eval(args)
    dist.barrier()
    rank_print("Done!")