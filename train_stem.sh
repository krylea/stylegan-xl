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

RES=$1

BATCH=$((BATCH_PER_GPU * SLURM_GPUS_ON_NODE)) 
GPUS=$SLURM_GPUS_ON_NODE
CPUS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS_ON_NODE))

python train.py --outdir=./training-runs/pokemon --cfg=stylegan3-t --data=./data/pokemon$RES.zip --dataset_name pokemon \
    --gpus=$GPUS --batch=$BATCH --mirror=1 --snap 10 --batch-gpu $BATCH_PER_GPU \
     --kimg 10000 --cbase 16384 --cmax 256 --syn_layers 7 --workers $CPUS --resolution $RES 


