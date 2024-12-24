#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -p p3 #partition
#SBATCH --nodelist cnode7-015
#SBATCH --cpus-per-task=8
cd /mnt/home/kaipoc/research_vh/LLaVA-NeXT

srun bash scripts/pred/dream1k_eval_parallel.sh