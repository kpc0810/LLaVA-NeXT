# Standard library imports
import argparse
import asyncio
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

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoConfig

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
from urllib.parse import urlparse, parse_qs

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
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    # parser.add_argument("--video_path", help="Path to the video files.", required=True)
    parser.add_argument("--data-file", help="Path to the data file.", required=False)
    parser.add_argument("--video_folder", help="Path to the video files.", required=False)
    parser.add_argument("--output-dir", help="Directory to save the model results JSON.", required=False)
    parser.add_argument("--output-name", help="Name of the file for storing results JSON.", required=False)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--frames_upbound", type=int, default=64)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    parser.add_argument("--captioning", action="store_true")
    
    return parser.parse_args()

def load_video(video_path,args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    try:
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=0)
    except:
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)

    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()

    return spare_frames,frame_time,video_time


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames

def merge_temp_files(output_dir, output_name, answers_path):
    # only the master process will merge the temp files to the final caption file
    if dist.get_rank() == 0:
        caption_file = open(answers_path, "a")
        all_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
        pattern = re.compile(rf"{re.escape(output_name)}\.temp\.\d+\.jsonl")
        all_temp_files = [f for f in all_files if pattern.search(os.path.basename(f))]
        for temp_file in all_temp_files:
            with open(temp_file, 'r') as temp_f:
                lines = temp_f.readlines()
                rank_print(f"== There are total {len(lines)} lines in {temp_file} ==")
                for line in lines:
                    caption_file.write(line)
            caption_file.flush()
            rank_print(f"== Finish merging {temp_file} into {answers_path} ==")
            os.remove(temp_file)
        caption_file.close() 

    return

def _timeout_handler(signum, frame):
    raise Exception("Function execution exceeded 30 minutes.")

def sample_from_vid_with_time_control(video_file, data_args):
    # set timeout for 30 minutes
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(1800)

    try:
        # execute long-time task
        result = process_video_with_decord(video_file, data_args)
        # close the alarm after the task is executed
        signal.alarm(0)
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        # timeout then return None
        return None

def run_inference(args):
    """
    Run inference on Miradata Captioning DataSet using the LLaVA-Video model.

    Args:
        args: Command-line arguments.
    """
    # Intialize the distributed environment
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)  # This sets the current GPU device to the one corresponding to the local rank
    
    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        print(f"model_name: {model_name}")
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        rank_print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
            
            overwrite_config.update({'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064})  # for solving the size mismatch error (model.model.embed_tokens.weight and model.lm_head)

            tokenizer, model, image_processor, context_len = load_pretrained_faith_model(args.model_path, model_name, load_8bit=args.load_8bit, torch_dtype="bfloat16", overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_faith_model(args.model_path, model_name, torch_dtype="bfloat16")
    else:
        pass
    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = True

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False
    DecordArgs = namedtuple('DecordArgs', ['video_fps', 'frames_upbound', 'force_sample']) 
    data_args = DecordArgs(video_fps=1, frames_upbound=args.frames_upbound, force_sample=args.force_sample)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the whole data, cached caption file, and set output path
    raw_data = json.load(open(args.data_file, 'r'))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    answers_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    if os.path.exists(answers_path):
        answers_file = open(answers_path, "r")
        cached_data = answers_file.readlines()
        answers_file.close()
        cached_data = [json.loads(line)["video"] for line in cached_data]
    else:
        os.makedirs(os.path.dirname(answers_path), exist_ok=True)
        open(answers_path, "w").close()
        cached_data = []
    # cache_set = set([f"{json.loads(line)['video_id']}-{json.loads(line)['clip_id']}" for line in cached_data])
    data = [sample for sample in raw_data if sample["video"] not in cached_data]
    rank_print(f"Loaded {len(data)} samples from {args.data_file}")
    data = get_chunk(data, world_size, local_rank)
    
    # Temp file for each process
    temp_answers_file = os.path.join(args.output_dir, f"{args.output_name}.temp.{local_rank}.jsonl")
    temp_answers_file = open(temp_answers_file, "w")
    cnt = 0
    # rounds = ['dense', 'main_object', 'background']
    sample_set = []
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    rank_print(f"=== Process {local_rank}: the length of data is {len(data)} ===")
    for i in tqdm(range(0, len(data))):
        cnt += 1
        sample = data[i]

        video_id = sample["video"]
        video_file = os.path.join(args.video_folder, f"{video_id}.mp4")
        rank_print(f"Captioning from video {video_id}, video name: {video_file}")
        
        if not os.path.exists(video_file):
            rank_print(f"Video file {video_file} does not exist")
            continue
        # video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, data_args)
        result = sample_from_vid_with_time_control(video_file, data_args)
        if result is None:
            continue
        video, video_time, frame_time, num_frames_to_sample = result
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda(local_rank)
        video = [video]

        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        
        prompt, question = "", ""
        if args.captioning:
            captioning_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\nPlease describe this video.'
            conv = copy.deepcopy(conv_templates[args.conv_mode])
            conv.append_message(conv.roles[0], captioning_prompt)
            conv.append_message(conv.roles[1], None)
            captioning_prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(captioning_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda(local_rank)
            if tokenizer.pad_token_id is None:
                if "qwen" in tokenizer.name_or_path.lower():
                    rank_print("Setting pad token to bos token for qwen model.")
                    tokenizer.pad_token_id = 151643
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda(local_rank)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            output_ids = model.module.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=1024 ,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
            video_caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().rstrip(stop_str).strip()
            
            prompt = "You are provided with a video, a pre-generated caption that describes the video and a set of several captions. " \
                     "Your task is to carefully watch the video, consider the pre-generated caption provided, " \
                     "and select the caption from the options that best aligns with both the video and the pre-generated caption. " \
                     "Provide your answer only as a single letter representing the option whose caption best describes the video, without any explanation." \
                     f"\nThe video's pre-generated caption is listed below:\n{video_caption}"
                     
            question = "Watch the video provided, refer to the pre-generated caption, " \
                       "and choose the option whose caption most accurately describes the video while aligning with the pre-generated caption.\n"
        else:
            prompt = "You are provided with a video and a set of several captions. " \
                     "Your task is to watch the video provided carefully, and select the caption that best describes the video. " \
                     "Provide your answer only as a single letter representing the option whose caption that best describes the video, without any explanation."
            question = "Watch the video provided, and choose the option whose caption describes the video most accurately.\n"
        
        key_order = list(sample["captions"].keys())
        random.shuffle(key_order)
        
        options = []
        ans = None
        for idx, key in enumerate(key_order):
            option_letter = chr(65 + idx)  # A, B, C etc.
            options.append(f"{option_letter}. {sample['captions'][key]}")
            if key == "1":  # Track correct answer
                ans = option_letter
        sample["answer"] = ans
        sample["captions"] = options
                
        # Combine into final prompt
        question += "\n".join(options)
        full_prompt = f"{prompt}\n{question}\nThe best answer is:"
        sample["prompt"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{full_prompt}'

        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], sample["prompt"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda(local_rank)
        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                rank_print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda(local_rank)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids = model.module.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=1024 ,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
        outputs = ""
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().rstrip(stop_str).strip()
        
        # save prediction and gt captions, while pop the unneeded elements
        sample["prediction"] = outputs
        
        sample_set.append(sample)
        temp_answers_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        temp_answers_file.flush()
        rank_print(f"=== Process {local_rank}: finish writing {cnt} processed ===")
        
        if cnt % 10 == 0:
            temp_answers_file.close()
            dist.barrier()
            merge_temp_files(args.output_dir, args.output_name, answers_path)
            dist.barrier()
            # create new temp file's' again
            temp_answers_file = os.path.join(args.output_dir, f"{args.output_name}.temp.{local_rank}.jsonl")
            temp_answers_file = open(temp_answers_file, "w")
    
    # Finish, then merge the temp file and write to the final file 
    temp_answers_file.close()
    dist.barrier()
    merge_temp_files(args.output_dir, args.output_name, answers_path)
    rank_print("Finish writing!")
    dist.barrier()
    return


if __name__ == "__main__":
    args = parse_args()
    
    run_inference(args)
    dist.barrier()
    rank_print("Done!")