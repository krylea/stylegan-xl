#!/bin/bash
#SBATCH --job-name=stylegan-xl
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB

BATCH_PER_GPU=16


python train.py --outdir=./training-runs/pokemon --cfg=stylegan3-t --data=./data/pokemon16.zip \
    --gpus=$SLURM_GPUS --batch=$BATCH_PER_GPU * $SLURM_GPUS --mirror=1 --snap 10 --batch-gpu $BATCH_PER_GPU \
     --kimg 10000 --cbase 16384 --cmax 256 --syn_layers 7 --workers $SLURM_CPUS_PER_GPU * $SLURM_GPUS


