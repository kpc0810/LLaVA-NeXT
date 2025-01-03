import pysubs2
import cv2
import numpy as np
import os
import re
import shutil


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def create_frame_output_dir(output_dir):
    """
    Create the output directory for storing the extracted frames.

    Parameters:
    output_dir (str): Path to the output directory.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        
def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def slice_frames(video_path, srt_path, frame_num, output_path):
    
    if not os.path.exists(srt_path):
        print(f"Subtitle file not found: {srt_path}")
        return
    
    print(f"Extracting video: {video_path}")
    subtitle_by_frame, total_frame = extract_subtitles(video_path, srt_path)
    if frame_num == -1:
        frame_num = total_frame
    uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

    subtitle_by_frame_idx = []
    for frame_idx in uniform_sampled_frames:
        for idx, title in enumerate(subtitle_by_frame):
            if frame_idx < title[1] and frame_idx >= title[0]:
                subtitle_by_frame_idx.append(idx)
    subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

    textlist = []
    for idx in subtitle_by_frame_idx:
        pattern = r'<font color="white" size=".72c">(.*?)</font>'
        raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
        try:
            textlist.append(raw_text[0])
        except:
            continue
    subtitle_text = "\n".join(textlist)
    
    video_id = os.path.basename(video_path).split(".")[0]
    # breakpoint()
    with open(f"{output_path}/{video_id}.txt", "w", encoding="utf-8") as f:
        f.write(subtitle_text)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video and corresponding subtitles.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--srt_path", type=str, default=None, help="Path to the subtitles file.")
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames to extract.")
    parser.add_argument("--output_path", type=str, default="output", help="Path to the output directory.")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    slice_frames(args.video_path, args.srt_path, args.num_frames, args.output_path)