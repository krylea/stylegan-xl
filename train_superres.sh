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

PREFIX=$1
RES=$2
PREV_RES=$((RES / 2))

BATCH=$((BATCH_PER_GPU * SLURM_GPUS_ON_NODE)) 
GPUS=$SLURM_GPUS_ON_NODE
CPUS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS_ON_NODE))

DONE=0
while [ $DONE -eq 0 ]
do
    DONE=1
    python train.py --outdir=./training-runs/pokemon --cfg=stylegan3-t --data=./data/pokemon$RES.zip \
        --gpus=$SLURM_GPUS_ON_NODE --batch=$BATCH --mirror=1 --snap 10 \
        --batch-gpu $BATCH_PER_GPU --kimg 10000 --syn_layers 7 --workers $CPUS \
        --superres --up_factor 2 --head_layers 4 --cbase 16384 --cmax 256 --restart_every 36000 \
        --path_stem training-runs/pokemon/$PREFIX-stylegan3-t-pokemon$PREV_RES-gpus$GPUS-batch$BATCH/best_model.pkl 
    if [[ $? -eq 3 ]]
    then
        DONE=0
    fi
done

