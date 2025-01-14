import argparse
import json
import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict, Counter
from functools import partial
from itertools import chain
from multiprocessing import Pool
from math import log
from decord import VideoReader, cpu
# import decord

# decord.bridge.set_bridge('torch')

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from FactVC import clip
from FactVC.emscore.scorer import EMScorer


def load_video_decord(video_path, fps=None):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=0)
    except:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_num = len(vr)
    fps = vr.get_avg_fps() if fps is None else fps

    frame_indices = np.arange(0, frame_num, int(fps))
    frames = vr.get_batch(frame_indices).asnumpy()
    
    pil_frames = []
    for frame in frames:
        frame_np = frame.astype(np.uint8)
        pil_frame = Image.fromarray(frame_np)
        pil_frames.append(pil_frame)
        
    return pil_frames


def load_video(video_path, fps=None):
    
    valid_video_path = None
    for ext in [".mp4", ".mkv", ".webm"]:
        valid_video_path = video_path + ext
        if os.path.exists(valid_video_path):
            return load_video_decord(valid_video_path, fps)
            
    return None


def parse_sent(sent):
    res = re.sub('[^a-zA-Z]', ' ', sent)
    res = res.strip().lower().split()
    return res


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = tokenizer(a, context_length=512, truncate=True)[0].tolist()
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})

    return idf_dict


def get_caption(args):
    
    """Obtain ground-truth captions and model-generated captions"""
    vids = [line.strip() for line in open(f'{args.data_dir}/{args.dataset}/vids.txt')]
    gt_caption, model_caption = [], []
    pred = {}
    if args.dataset == 'activitynet':
        gt_cap_files = [f'{args.data_dir}/{args.dataset}/captions/gt_ae_test_1_para.json',
                        f'{args.data_dir}/{args.dataset}/captions/gt_ae_test_2_para.json']
        gt_cap_jsons = [json.load(open(fn)) for fn in gt_cap_files]
        for vid in vids:
            cur_refs = [' '.join(parse_sent(gt_cap_jsons[0][vid]))]
            if vid in gt_cap_jsons[1]:
                cur_refs.append(' '.join(parse_sent(gt_cap_jsons[1][vid])))
            else:
                cur_refs.append(' '.join(parse_sent(gt_cap_jsons[0][vid])))
            gt_caption.append(cur_refs)
        
        with open(args.pred_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                pred[sample["idx"]] = sample["prediction"].strip()
                
        model_caption = []
        for vid in vids:
            model_caption.append(pred[vid])

    else:
        gt_cap_json = json.load(open(f'{args.data_dir}/{args.dataset}/captions/gt_val_para.json'))
        for vid in vids:
            gt_caption.append([' '.join(parse_sent(gt_cap_json[vid]))])
        
        with open(args.pred_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                pred[sample["idx"]] = sample["prediction"].strip()  
        for vid in vids:
            model_caption.append(pred[vid])

    # gt_caption_flatten = gt_caption * 6
    # model_caption_flatten = []
    # for cap_list in model_caption:
    #     model_caption_flatten.extend(cap_list)

    return gt_caption, model_caption


def get_factvc_score(args, gt_caption, model_caption):
    """Compute FactVC scores"""
    clip_model_checkpoints = os.path.join(os.path.abspath(args.data_dir), "..", "pretrained_models/factvc_video.pth")
    clip_model, transform = clip.load(clip_model_checkpoints, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using clip model {clip_model_checkpoints}')
    vids = [line.strip() for line in open(f'{args.data_dir}/{args.dataset}/vids.txt')]
    vid_feat_dict = {}
    
    for vid in tqdm(vids):
        
        video_path = f"{args.data_dir}/{args.dataset}/raw_videos/{vid}"
        video = load_video(video_path)
        video_inputs = []
        for frame in video:
            video_inputs.append(transform(frame))
        video_inputs = torch.stack(video_inputs).cuda()
        with torch.no_grad():
            video_features = clip_model.encode_image(video_inputs)
            video_features /= video_features.norm(dim=-1, keepdim=True)
            vid_feat_dict[vid] = video_features.cpu()

    # prepare idf
    corpus_json = json.load(open(f'{args.data_dir}/{args.dataset}/captions/ref_paragraphs.json'))
    corpus = []
    for _, sent in corpus_json.items():
        corpus.append(' '.join(parse_sent(sent[0])))
    idf_dict = get_idf_dict(corpus, clip.tokenize, nthreads=4)
    metric = EMScorer(vid_feat_cache=vid_feat_dict, clip_model=clip_model_checkpoints)
    results = metric.score(cands=model_caption, refs=gt_caption, vids=vids, idf=idf_dict,
                           cogr_weight=0.25, ref_weight=0.5, batch_size=64)
    return results


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Compute correlation between FactVC and factuality annotation')
#     parser.add_argument('--pred_file', type=str, default="outputs/factvc/pred_result/llava_video_test_0104_1.jsonl")
#     parser.add_argument('--data_dir', type=str, default="playground/FactVC/data")
#     parser.add_argument('--dataset', type=str, default='activitynet', choices=['activitynet', 'youcook2'])
#     args = parser.parse_args()

#     gt_cap, model_cap = get_caption(args)
#     factvc_score = get_factvc_score(args, gt_cap, model_cap)
