#!/bin/bash
#SBATCH --job-name=stylegan-xl
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=100GB

BATCH_PER_GPU=16

RES=$1
DATASET_NAME=$2
ckpt=${3:-''}
kimg=${4:-10000}

if [ -z $SLURM_CPUS_PER_GPU ]
then
    SLURM_CPUS_PER_GPU=1
fi
if [ -z $SLURM_GPUS_ON_NODE ]
then
    SLURM_GPUS_ON_NODE=1
fi

BATCH=$((BATCH_PER_GPU * SLURM_GPUS_ON_NODE)) 
GPUS=$SLURM_GPUS_ON_NODE
#GPUS=1
CPUS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS_ON_NODE))
#CPUS=1
#BATCH=$BATCH_PER_GPU


if [[ $DATASET_NAME == 'imagenet' ]]
then
    argstring="--outdir=./training-runs/$DATASET_NAME --cfg=stylegan3-t --data=/scratch/hdd001/datasets/imagenet/train --dataset_name $DATASET_NAME \
        --gpus=$GPUS --batch=$BATCH --mirror=1 --snap 30 \
        --batch-gpu $BATCH_PER_GPU --kimg $kimg --syn_layers 10 --workers $CPUS \
        --resolution $RES"

else
    argstring="--outdir=./training-runs/$DATASET_NAME --cfg=stylegan3-t --data=./data/${DATASET_NAME}${RES}.zip --dataset_name $DATASET_NAME \
        --gpus=$GPUS --batch=$BATCH --mirror=1 --snap 10 \
        --batch-gpu $BATCH_PER_GPU --kimg $kimg --syn_layers 7 --workers $CPUS \
        --cbase 16384 --cmax 256 --resolution $RES" 
fi

if [ -z $ckpt ]
then
    argstring="$argstring --resume $ckpt"
fi

python train.py $argstring


