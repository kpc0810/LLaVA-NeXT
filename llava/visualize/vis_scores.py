import os
import json
import glob
import matplotlib.pyplot as plt

def plot_metric(metric_name, scores, output_dir, exp_name):
    plt.figure(figsize=(16, 8))  # Set the figure size to double the original length
    x_values = list(scores.keys())
    y_values = list(scores.values())
    plt.plot(x_values, y_values, marker='o')
    plt.title(f'{metric_name} over {exp_name}')
    plt.xlabel('Checkpoint Iteration')
    plt.ylabel(metric_name)
    plt.grid(True)
    
    # Annotate each point with its score
    for x, y in zip(x_values, y_values):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center',
                     fontsize=8, rotation=45)
    
    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'))
    plt.close()

def main(exp_name):
    score_data_dir = os.path.join("outputs/miradata/scores/", exp_name)
    score_data_paths = glob.glob(os.path.join(score_data_dir, "score_*.json"))
    score_data_paths = sorted(score_data_paths, key=lambda x: int(x.split('_')[-1].split('.')[0].strip('[]')))
    
    all_scores = {}
    for score_data_path in score_data_paths:
        score_file_name = os.path.basename(score_data_path)
        ckpt_iter = int(score_file_name.split('_')[-1].split('.')[0].strip('[]'))
        with open(score_data_path, "r") as f:
            score_data = json.load(f)
        all_scores[ckpt_iter] = score_data

    dense_vis_dir = os.path.join("outputs/miradata/vis_results/", exp_name, "dense")
    main_obj_vis_dir = os.path.join("outputs/miradata/vis_results/", exp_name, "main_obj")
    background_vis_dir = os.path.join("outputs/miradata/vis_results/", exp_name, "background")
    
    os.makedirs(dense_vis_dir, exist_ok=True)
    os.makedirs(main_obj_vis_dir, exist_ok=True)
    os.makedirs(background_vis_dir, exist_ok=True)
    
    # Plotting metrics for each category
    for category, vis_dir in zip(['dense', 'main_object', 'background'], [dense_vis_dir, main_obj_vis_dir, background_vis_dir]):
        category_scores = {ckpt_iter: data[category] for ckpt_iter, data in all_scores.items()}
        for metric in category_scores[next(iter(category_scores))].keys():
            metric_scores = {ckpt_iter: scores[metric] for ckpt_iter, scores in category_scores.items()}
            plot_metric(metric, metric_scores, vis_dir, exp_name)

if __name__ =="__main__":
    exp_name = "llava-qwen-7b_fcl-ehn_nnode=32_epo=2_plmlr=5e-6_vtlr=1e-6_bspd=1_cplr=1e-4_aslr=1e-4_vcclwt=0.5_tpoclwt=0.25_tpaclwt=0.25"  # This is the only one param need to be changed
    main(exp_name)
    print(f"Done for {exp_name}!")