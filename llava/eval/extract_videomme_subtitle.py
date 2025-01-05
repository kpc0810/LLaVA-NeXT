import os
import argparse
import subprocess
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video and corresponding subtitles.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--srt_path", type=str, default=None, help="Path to the subtitles file.")
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames to extract.")
    parser.add_argument("--output_path", type=str, default="playground/videomme/subtitle_txt", help="Path to the output directory.")
    
    return parser.parse_args()


def main(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    for video in tqdm(os.listdir(args.video_path)):
        
        video_path = os.path.join(args.video_path, video)
        srt_path = os.path.join(args.srt_path, video.replace(".mp4", ".srt"))
        
        subprocess.run([
            "python3", "llava/eval/slice_and_extract.py",
            "--video_path", video_path,
            "--srt_path", srt_path,
            "--num_frames", str(args.num_frames),
            "--output_path", args.output_path
        ])
        
if __name__ == "__main__":
    args = parse_args()
    main(args)