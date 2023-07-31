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

PREFIX=$1
RES=$2
DATASET_NAME=$3
PREV_RES=$((RES / 2))
ckpt=${4:-''}
kimg=${5:-10000}
desc=${6:-''}

if [[ -z $SLURM_CPUS_PER_GPU ]]
then
    SLURM_CPUS_PER_GPU=1
fi
if [[ -z $SLURM_GPUS_ON_NODE ]]
then
    SLURM_GPUS_ON_NODE=1
fi

BATCH=$((BATCH_PER_GPU * SLURM_GPUS_ON_NODE)) 
GPUS=$SLURM_GPUS_ON_NODE
CPUS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS_ON_NODE))



if [[ $DATASET_NAME == 'imagenet' ]]
then
    argstring="--outdir=./training-runs/$DATASET_NAME --cfg=stylegan3-t --data=/scratch/hdd001/datasets/imagenet/train --dataset_name $DATASET_NAME \
            --gpus=$SLURM_GPUS_ON_NODE --batch=$BATCH --mirror=1 --snap 30 \
            --batch-gpu $BATCH_PER_GPU --kimg $kimg --syn_layers 10 --workers $CPUS \
            --superres --up_factor 2 --head_layers 7 --restart_every 36000 --resolution $RES \
            --path_stem training-runs/$DATASET_NAME/$PREFIX-stylegan3-t-${DATASET_NAME}${PREV_RES}-gpus1-batch16/best_model.pkl"

else
    argstring=" --outdir=./training-runs/$DATASET_NAME --cfg=stylegan3-t --data=./data/${DATASET_NAME}${RES}.zip --dataset_name $DATASET_NAME \
            --gpus=$SLURM_GPUS_ON_NODE --batch=$BATCH --mirror=1 --snap 10 \
            --batch-gpu $BATCH_PER_GPU --kimg 10000 --syn_layers 7 --workers $CPUS \
            --superres --up_factor 2 --head_layers 4 --cbase 16384 --cmax 256 --restart_every 36000 --resolution $RES \
            --path_stem training-runs/$DATASET_NAME/$PREFIX-stylegan3-t-${DATASET_NAME}${PREV_RES}/best_model.pkl"
fi

if [[ -n $ckpt ]]
then
    argstring="$argstring --resume $ckpt"
fi

if [[ -n $desc ]]
then
    argstring="$argstring --desc $desc"
fi

DONE=0
while [ $DONE -eq 0 ]
do
    DONE=1 
    python train.py $argstring
    if [[ $? -eq 3 ]]
    then
        DONE=0
    fi
done

