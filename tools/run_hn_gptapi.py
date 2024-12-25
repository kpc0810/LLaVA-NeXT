import os
import sys
import re
import csv
from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm
import backoff
from openai import RateLimitError

# OpenAI API key
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6IjY0Y2VjNzNjLWYyNjUtNGIxNy1hMGEzLTljMTU5ZDBhODQ2NCIsInNlY3JldCI6IkJFYi9IeGdVR0hqSWF1VW50TFM4TjJTd2xQRFZBZWZqVUZ6dmprSW1XVjQ9In0.GFMBJlW37Oxu57ZRuLnJd4LsZSY_3gAn_9Xc6VaYCfM"


def load_csv(file_path):
    return pd.read_csv(file_path)


def extract_captions(df):
    return {
        'dense': df['dense_caption'].tolist(),
        'main_object': df['main_object_caption'].tolist(),
        'background': df['background_caption'].tolist()
    }


@backoff.on_exception(backoff.expo, RateLimitError)
def hallucinate_caption(caption):
    prompt = f"Input: {caption}\nOutput:"
    
    client = AzureOpenAI(
        api_version="2024-02-15-preview",
        azure_endpoint="https://llm-proxy.perflab.nvidia.com",
        api_key=API_KEY,
    )
    
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": (
                "Hallucination in Large-scale Visual Language Models (LVLMs) refers to cases where these models generate descriptions introducing elements that are inconsistent with the content or completely absent from a provided image. These hallucinations can be coarse-grained, focusing on the mere existence of objects, or fine-grained, focusing on more specific attributes or characteristics such as quantity, properties, and locations. It’s noteworthy that LVLMs often hallucinate about objects frequently present in visual instructions or within the actual image contents. Your task is to revise a given caption to create a mirrored version that closely aligns with the original’s content and length but incorporates elements of hallucination. Please craft two versions of this ’hallucinated’ caption, each one representing either coarse-grained or fine-grained hallucinations. The first step involves identifying the objects involved and their associated attributes within the given caption. Subsequently, combine this insight with the details concerning hallucinations provided above to complete your task."
                "Input: A woman stands in the dining area at the table."
                "Output: A woman sitting in the classroom in front of the blackboard"
                "===\n"
                "Input: A room with chairs, a table, and a woman in it."
                "Output: A room with a fireplace, a computer and a man in it"
                "===\n"
                "Input: The large brown bear has a black nose."
                "Output: The cut dog owns a brown nose and blue eyes"
                "===\n"
                "Input: The large brown bear has a black nose."
                "Output: The cut dog owns a brown nose and blue eyes"
                "===\n"
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=128,
    )
    
    return response.choices[0].message.content

def prompt_and_save_captions(input_file_path, output_file_path):
    
    # Create cache_set to collect the ids of already processed data
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            reader = csv.DictReader(f)
            processed_clip_ids = set(row['clip_id'] for row in reader)
    else:
        processed_clip_ids = set()
    
    # Read data and fieldnames, skip processed data
    data, fieldnames = [], []
    with open(input_file_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['clip_id'] not in processed_clip_ids:
                data.append(row)
    fieldnames += ['dense_hallucinated_caption', 'main_object_hallucinated_caption', 'background_hallucinated_caption']
    
    # Create the output file
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Process data
    for row in tqdm(data):
        for category in ['dense', 'main_object', 'background']:
            caption = row[f'{category}_caption']
            hallucinated_caption = hallucinate_caption(caption)
            row[f'{category}_hallucinated_caption'] = hallucinated_caption.replace('\n', ' ').strip()

        with open(output_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows([row])
    
    print(f"Generation of hallucination captions is completed. Results saved to {output_file_path}.", flush=True)

if __name__ == "__main__":
    """
    python run_hn_gptapi.py $train_split_idx
    """

    ## train split paths
    input_file_path = f'/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/fixed_parsed_data/merged_miradata_84k_train_dataset.csv'
    output_file_path = f'/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/hn_data/merged_miradata_84k_train_dataset.csv'
    log_file_path = f'/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/tools/hn/merged_miradata_84k_train_dataset.txt'
    
    ## debug
    # input_file_path = '/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/miradata_v1_100_samples.csv'
    # output_file_path = '/home/kaipoc/personal/research_vh/VILA/playground/data/eval/miradata/hn_data/miradata_v1_100_samples.csv'
    # log_file_path = '/home/kaipoc/personal/research_vh/LLaVA-NeXT/slurm_log/tools/hn/miradata_v1_100_samples.txt'
    
    logfile = open(log_file_path, 'w')
    sys.stdout = logfile
    
    prompt_and_save_captions(input_file_path, output_file_path)
    
    logfile.close()