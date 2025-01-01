import os
import re
import ast
import csv
import copy
import json
from typing import Dict, Sequence
import torch
import random

from llava.datamodule.lazy import LazySupervisedDataset, DataCollatorForSupervisedDataset, preprocess_multimodal, preprocess
from llava.utils import rank0_print, process_video_with_decord, rank0_breakpoint
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.datamodule.mira_user_prompt import user_prompts
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.utils import find_most_similar_substring, get_phrase_indices
from llava import conversation as conversation_lib

from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet



class MiraHaclContrastiveDataset(LazySupervisedDataset):
    def __init__(self, tokenizer, data_path, data_args, model_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.list_data_dict = self.get_data(data_path)
        assert self.data_args.add_time_instruction == True, "curretly only support LLaVA-Video"
        
    def get_data(self, data_path):
        with open(data_path, 'r') as csvfile:
            data = [json.loads(json.dumps(row)) for row in csv.DictReader(csvfile)]
            print(f"Loaded {len(data)} rows from {data_path}")
        
        return_data = []
        for i, sample in enumerate(data):
            sample["video"] = os.path.join(self.data_args.video_folder, sample["file_path"])
            if not os.path.exists(sample["video"]):
                continue
            gt_captions = {
                'dense': sample.pop('dense_caption'),
                'main_object': sample.pop('main_object_caption'),
                'background': sample.pop('background_caption')
            }
            hallu_captions = {
                'dense': sample.pop('dense_hallucinated_caption'),
                'main_object': sample.pop('main_object_hallucinated_caption'),
                'background': sample.pop('background_hallucinated_caption')
            }
            all_caption_types = ['dense', 'main_object', 'background']
            for caption_type in all_caption_types:
                sample['caption_type'] = caption_type
                sample["conversations"] = [
                    {"from": "human", "value": random.choice(user_prompts[caption_type])},
                    {"from": "gpt", "value": gt_captions[caption_type]}
                ]
                sample["hallu_conversations"] = [
                    {"from": "human", "value": random.choice(user_prompts[caption_type])},
                    {"from": "gpt", "value": hallu_captions[caption_type]}
                ]
                return_data.append(sample.copy())
        return return_data

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        
        ## id, input_ids, labels, image
        clip_id = sample["clip_id"]
        
        video_file = sample["video"]
        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)
        assert num_frames_to_sample == self.data_args.frames_upbound, f"num_frames_to_sample:{num_frames_to_sample} is less than frames_upbound:{self.data_args.frames_upbound}."
        
        processor = self.data_args.image_processor
        image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        sample["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sample["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
        image = [(image, video[0].size, "video")]
        proc_conversations = preprocess_multimodal(copy.deepcopy([sample["conversations"]]), self.data_args)
        has_image = True  # have processed; otherwise, num_frames_to_sample != self.data_args.frames_upbound
        data_dict = preprocess(proc_conversations, self.tokenizer, has_image=has_image)
        
        data_dict = {
            "id": clip_id,
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "image": image
        }
        
        ## hallu_input_ids (completed hallucinated caption)
        caption = proc_conversations[0][1]["value"]
        sample["hallu_conversations"][0]["value"] = f'{time_instruciton}\n{sample["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
        proc_conversations = preprocess_multimodal(copy.deepcopy([sample["conversations"]]), self.data_args)
        hallu_data_dict = preprocess(proc_conversations, self.tokenizer, has_image=has_image)
        data_dict["hallu_input_ids"] = hallu_data_dict["input_ids"][0]

        return data_dict

    def __len__(self):
        return len(self.list_data_dict[:128145]) # Hard code for fair comparision


class MiraHaclDataCollatorForContrastiveDataset(DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        
        # add for contrastive learning
        hallu_input_ids = []
        for instance in instances:
            # for hallucination aug
            hallu_input_ids.append(instance["hallu_input_ids"])
        
        # pad
        hallu_input_ids = self.pad_sequence(hallu_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # truncate
        hallu_input_ids = hallu_input_ids[:, : self.tokenizer.model_max_length]
        
        batch.update({
            "hallu_input_ids": hallu_input_ids
        })
        
        return batch