## DREAM-1K
### Download
```
git lfs install
git clone https://huggingface.co/datasets/omni-research/DREAM-1K
```

### Extra Packages
* update transformers (for llama3)
* huggingface_hub (login)

### run full code (eval + score)
```
bash scripts/pred/dream1k.sh
```

* modify the argument in scripts/pred/dream1k.sh or or change the argument into $1, $2, ...
```
nproc_per_node=2
data_path="DREAM-1K/json/metadata.json"
video_folder="DREAM-1K/video/DREAM-1K_videos"
output_dir="outputs/dream1k"
output_name="dream1k_pred_results"
huggingface_token=""
```
* Change the export PATH to your env
```
export PATH=/mnt/home/kaipoc/miniconda3/envs/llava/bin:$PATH
```

### compute score only
```
bash scripts/pred/dream1k_eval.sh <pred_path> <huggingface_token>
```
* Change the export PATH to your env mentioned above

### directly login huggingface
```
export PATH=/mnt/home/kaipoc/miniconda3/envs/llava/bin:$PATH
huggingface-cli login
enter your huggingface token
```
