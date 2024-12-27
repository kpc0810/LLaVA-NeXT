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
# nltk.download('wordnet')

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


class MiraLazyContrastiveDataset(LazySupervisedDataset):
    def __init__(self, tokenizer, data_path, data_args, model_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.list_data_dict = self.get_data(data_path)
        assert self.data_args.add_time_instruction == True, "curretly only support LLaVA-Video"
        
        # stemmer and similarity model
        self.stemmer = PorterStemmer()
        self.similarity_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
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

    def _find_synonyms_hypernyms(self, target_word):
        synonyms, hypernyms = set(), set()
        for synset in wordnet.synsets(target_word):
            for word in synset.hypernyms():
                word = word.name().split(".")[0]
                word = word.replace("_", " ")
                hypernyms.add(word)
            for word in synset.lemmas():
                word = word.name().replace("_", " ")
                synonyms.add(word)
        
        return_words = []
        cand_words = list(synonyms) + list(hypernyms)
        for word in cand_words:
            try:
                similarity_score = self.similarity_model.similarity(word, target_word)
            except KeyError as e:
                similarity_score = 0.0
            if similarity_score > 0.3:
                return_words.append(word)
        
        return return_words

    def _add_rule_for_block_words(self, block_words):
        should_add_words = []
        for block_word in block_words:
            block_word = block_word.strip().lower()
            # human, person, character
            if ("man" in block_word) or ("woman" in block_word) or ("child" in block_word):
                should_add_words += ["human", "person", "character"]
        return list(set(block_words + should_add_words))
    
    def get_block_ids(self, caption, sample):
        block_words = []
        for rel_pair in [sample['dense_relation_pair'], sample['main_object_relation_pair'], sample['background_relation_pair']]:
            for pair in eval(rel_pair):
                block_words += list(pair)
        
        # for more accurately matching the word in gt_caption, adjust the gt_seman_words
        adjusted_block_words = []
        for word in block_words:
            if word in caption:
                adjusted_block_words.append(word)
            else:
                stemmed_word = self.stemmer.stem(word)
                capitalized_word = word[0].upper() + word[1:]
                if stemmed_word in caption:
                    pattern = r'\b' + re.escape(stemmed_word) + r'\w*\b'
                    match_words = re.findall(pattern, caption)
                    adjusted_block_words += match_words
                elif capitalized_word in caption:
                    adjusted_block_words.append(capitalized_word)
        
        block_words = list(set(adjusted_block_words))
        
        # augment the gt_seman_words by the synonyms and hypernyms of the gt_sem pan_words
        augmented_block_words = []
        for word in block_words:
            synonyms = self._find_synonyms_hypernyms(word)
            augmented_block_words += synonyms
        
        block_words += augmented_block_words
        block_words = list(set(block_words))
        
        # add the manual rule for block words
        block_words = self._add_rule_for_block_words(block_words)
        
        # convert the block_words to ids
        block_ids = []
        for word in block_words:
            if self.tokenizer.__class__.__name__ == 'Qwen2TokenizerFast':
                word = f" {word}"
                block_ids += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            else:
                raise NotImplementedError(f"Unsupported tokenizer: {self.tokenizer.__class__.__name__}")
        
        return block_ids

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
        
        ## block_ids
        caption = proc_conversations[0][1]["value"]
        block_ids = torch.tensor(self.get_block_ids(caption, sample), dtype=data_dict["input_ids"].dtype)
        data_dict["block_ids"] = block_ids
        
        ## hallu_input_ids
        sys_prompt = proc_conversations[0][0]["value"]
        conv_template = self.model_args.version
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], sys_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        hallu_input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        data_dict["hallu_input_ids"] = hallu_input_ids


        ## obj_phrase_indices and act_phrase_indices (Note that the `occupied_token_indices` are all appeared phrase indexes in `caption_ids`)
        caption_ids = self.tokenizer.encode(f"{caption}{self.tokenizer.eos_token}\n", add_special_tokens=False)
        caption_prefix_length = len(data_dict["input_ids"]) - len(caption_ids)

        # find the most similar obj words and act words to reconstruct the relation_pair
        relation_pair = ast.literal_eval(sample[f'{sample["caption_type"]}_relation_pair'])
        new_relation_pair = []
        for pair in relation_pair:
            new_relation_pair.append([find_most_similar_substring(caption, pair[0])] + [find_most_similar_substring(caption, word) for word in pair[1:]])
        relation_pair = new_relation_pair

        obj_words, act_words = [], []
        for pair in relation_pair:
            act_words += [pair[0]]
            obj_words += [word for word in pair[1:]]
        obj_words, act_words = list(set(obj_words)), list(set(act_words))
        obj_words, obj_phrase_indices = get_phrase_indices(self.tokenizer, obj_words, torch.tensor(caption_ids), caption_prefix_length)  # (List[List[int]]), [num_obj_words, occupied_token_indices]
        act_words, act_phrase_indices = get_phrase_indices(self.tokenizer, act_words, torch.tensor(caption_ids), caption_prefix_length)  # (List[List[int]]), [num_act_words, occupied_token_indices]
        

        ## obj_patch_indices (List[List[int]]): [num_obj_words, num_frames, occupied_patch_indices]
        occupied_patches = json.loads(sample['occupied_patches']) 
        # find the most similar obj words for each tracklet
        new_occupied_patches = {}  # Dict[int, Dict[str, List[int]]]: frame_idx -> obj_word -> obj_patches
        for frame_idx, frame_obj_patches in occupied_patches.items():
            new_frame_obj_patches = {}
            for obj_word, obj_patches in frame_obj_patches.items():
                new_frame_obj_patches[find_most_similar_substring(caption, obj_word)] = obj_patches
            new_occupied_patches[frame_idx] = new_frame_obj_patches
        occupied_patches = new_occupied_patches
        
        obj_patch_indices = []
        for obj_word in obj_words:
            obj_word = obj_word.strip()
            all_patches_of_obj_word = []
            for frame_idx, obj_patches in occupied_patches.items():
                if obj_word in obj_patches:
                    patches_of_obj_word = obj_patches[obj_word]
                else:
                    patches_of_obj_word = []
                all_patches_of_obj_word.append(patches_of_obj_word)

            # Append the [] until the length of all_patches_of_obj_word is `num_frames_to_sample`, for some case that the detected frames are less than `num_frames_to_sample`
            while len(all_patches_of_obj_word) < num_frames_to_sample:
                all_patches_of_obj_word.append([])
            obj_patch_indices.append(all_patches_of_obj_word)

        # remove the sample that no patches are detected in the video   
        self.exclude_no_detected_sample(obj_patch_indices)

        # remove the single empty tracklet indices and their corresponding phrase indices
        filtered_obj_patch_indices, filtered_obj_words, filtered_obj_phrase_indices = [], [], []
        for obj_word_idx in range(len(obj_patch_indices)):
            if obj_patch_indices[obj_word_idx] != [[] for _ in range(num_frames_to_sample)]:
                filtered_obj_patch_indices.append(obj_patch_indices[obj_word_idx])
                filtered_obj_words.append(obj_words[obj_word_idx])
                filtered_obj_phrase_indices.append(obj_phrase_indices[obj_word_idx])
        
        obj_patch_indices, obj_words, obj_phrase_indices = filtered_obj_patch_indices, filtered_obj_words, filtered_obj_phrase_indices
        data_dict["obj_words"] = obj_words
        data_dict["obj_phrase_indices"] = obj_phrase_indices
        data_dict["obj_patch_indices"] = obj_patch_indices


        ## act_patch_indices: (List[List[int]]): [num_act_words, num_frames, occupied_patch_indices]
        # only corresponded act patch that two related obj_patch_indices are detected should be included
        act2obj_map = { elem[0]: list(elem[1:]) for elem in relation_pair }
        act_patch_indices = []
        strip_obj_words = [obj_word.strip() for obj_word in obj_words]
        for act_word in act_words:
            act_word = act_word.strip()
            related_obj_words = [related_obj_word for related_obj_word in act2obj_map[act_word] if related_obj_word in strip_obj_words]
            all_patches_of_act_word = []
            for related_obj_word in related_obj_words:
                related_obj_word_idx = strip_obj_words.index(related_obj_word)
                if len(all_patches_of_act_word) != 0:
                    assert len(all_patches_of_act_word) == len(obj_patch_indices[related_obj_word_idx]) == num_frames_to_sample, \
                        f"length not equal: {len(all_patches_of_act_word)} != {len(obj_patch_indices[related_obj_word_idx])} != {num_frames_to_sample}!"
                    for frame_idx in range(len(all_patches_of_act_word)):
                        all_patches_of_act_word[frame_idx] += obj_patch_indices[related_obj_word_idx][frame_idx]
                        all_patches_of_act_word[frame_idx] = list(set(all_patches_of_act_word[frame_idx]))  # remove the duplicate patch indices
                else:
                    all_patches_of_act_word = copy.deepcopy(obj_patch_indices[related_obj_word_idx])
            # pad empty patch indices to the length of `num_frames_to_sample` if this act patch is not detected in any frame
            while len(all_patches_of_act_word) < num_frames_to_sample:
                all_patches_of_act_word.append([])
            act_patch_indices.append(all_patches_of_act_word)

        # remove the sample that no patches are detected in the video
        self.exclude_no_detected_sample(act_patch_indices)

        # remove the single empty patch indices and their corresponding phrase indices
        filtered_act_patch_indices, filtered_act_words, filtered_act_phrase_indices = [], [], []
        for act_word_idx in range(len(act_patch_indices)):
            if act_patch_indices[act_word_idx] != [[] for _ in range(num_frames_to_sample)]:
                filtered_act_patch_indices.append(act_patch_indices[act_word_idx])
                filtered_act_words.append(act_words[act_word_idx])
                filtered_act_phrase_indices.append(act_phrase_indices[act_word_idx])

        act_patch_indices, act_words, act_phrase_indices = filtered_act_patch_indices, filtered_act_words, filtered_act_phrase_indices
        
        data_dict["act_words"] = act_words
        data_dict["act_phrase_indices"] = act_phrase_indices
        data_dict["act_patch_indices"] = act_patch_indices
        return data_dict
    
    def exclude_no_detected_sample(self, patch_indices):
        """ Exclude the sample that no patches are detected in the video.
        """
        num_words = len(patch_indices)
        unwanted_patch_indices = [[[] for i in range(self.data_args.frames_upbound)] for j in range(num_words)]
        if patch_indices == unwanted_patch_indices:
            raise ValueError("No patches are detected in the video!")

class MiraDataCollatorForContrastiveDataset(DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        
        # add for contrastive learning
        hallu_input_ids, block_ids = [], []
        obj_patch_indices, obj_words, obj_phrase_indices = [], [], []
        act_patch_indices, act_words, act_phrase_indices = [], [], []
        for instance in instances:
            # for hallucination aug
            hallu_input_ids.append(instance["hallu_input_ids"])
            block_ids.append(instance["block_ids"])
            
            # for frame-phrase contrastive learning
            if isinstance(instance["obj_patch_indices"], list) \
                and isinstance(instance["obj_phrase_indices"], list) \
                and isinstance(instance["act_patch_indices"], list) \
                and isinstance(instance["act_phrase_indices"], list):
                obj_patch_indices.append(instance["obj_patch_indices"])
                obj_words.append(instance["obj_words"])
                obj_phrase_indices.append(instance["obj_phrase_indices"])
                act_patch_indices.append(instance["act_patch_indices"])
                act_words.append(instance["act_words"])
                act_phrase_indices.append(instance["act_phrase_indices"])
            else:
                raise ValueError("obj_patch_indices, obj_phrase_indices, and act_patch_indices, act_phrase_indices must be lists")
        
        # pad
        hallu_input_ids = self.pad_sequence(hallu_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # truncate
        hallu_input_ids = hallu_input_ids[:, : self.tokenizer.model_max_length]
        hallu_attention_mask = hallu_input_ids.ne(self.tokenizer.pad_token_id)
        
        batch.update({
            "hallu_input_ids": hallu_input_ids,
            "hallu_attention_mask": hallu_attention_mask,
            "block_ids": block_ids,
            "obj_patch_indices": obj_patch_indices,
            "obj_words": obj_words,
            "obj_phrase_indices": obj_phrase_indices,
            "act_patch_indices": act_patch_indices,
            "act_words": act_words,
            "act_phrase_indices": act_phrase_indices
        })
        
        return batch
