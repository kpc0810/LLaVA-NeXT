import os
import csv
import copy
import json
from typing import Dict
import torch
import random

from llava.datamodule.lazy import LazySupervisedDataset, DataCollatorForSupervisedDataset, preprocess_multimodal, preprocess
from llava.utils import rank0_print, process_video_with_decord
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.datamodule.mira_user_prompt import user_prompts

class MiraLazySupervisedDataset(LazySupervisedDataset):
    def __init__(self, tokenizer, data_path, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = self.get_data(data_path)
        
    def get_data(self, data_path):
        with open(data_path, 'r') as csvfile:
            data = [json.loads(json.dumps(row)) for row in csv.DictReader(csvfile)]
            print(f"Loaded {len(data)} rows from {data_path}")
        
        return_data = []
        for i, sample in enumerate(data):
            sample["video"] = os.path.join(self.data_args.video_folder, sample["file_path"])
            if not os.path.exists(sample["video"]):
                print("Video path {} is not exist!".format(sample["video"]))
                continue
            gt_captions = {
                'dense': sample.pop('dense_caption'),
                'main_object': sample.pop('main_object_caption'),
                'background': sample.pop('background_caption')
            }
            all_caption_types = ['dense', 'main_object', 'background']
            for caption_type in all_caption_types:
                sample['caption_type'] = caption_type
                sample["conversations"] = [
                    {"from": "human", "value": random.choice(user_prompts[caption_type])},
                    {"from": "gpt", "value": gt_captions[caption_type]}
                ]
                return_data.append(sample.copy())
        return return_data
    
    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        clip_id = sources["clip_id"]
        
        video_file = sources["video"]
        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)
        assert num_frames_to_sample == self.data_args.frames_upbound, f"num_frames_to_sample:{num_frames_to_sample} is less than frames_upbound:{self.data_args.frames_upbound}."
        
        processor = self.data_args.image_processor
        image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
        if self.data_args.add_time_instruction:
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
            sources["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
        image = [(image, video[0].size, "video")]
        
        sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
        
        has_image = True  # have processed; otherwise, num_frames_to_sample != self.data_args.frames_upbound

        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict["image"] = image
        data_dict["id"] = clip_id

        return data_dict
        
class MiraDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    pass