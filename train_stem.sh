#!/bin/bash
#SBATCH --job-name=stylegan-xl
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --nodes=1-2  # number of nodes
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --gres=gpu:4  # number of gpus per node
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=100GB

BATCH_PER_GPU=16

RES=$1
DATASET_NAME=$2
ckpt=${3:-''}
kimg=${4:-10000}
desc=${5:-''}
MASTER_PORT=$6

if [[ -z $SLURM_CPUS_PER_GPU ]]
then
    SLURM_CPUS_PER_GPU=1
fi
if [[ -z $SLURM_GPUS_ON_NODE ]]
then
    SLURM_GPUS_ON_NODE=1
fi
if [[ -z $SLURM_JOB_NUM_NODES ]]
then
    SLURM_JOB_NUM_NODES=1
fi

BATCH=$((BATCH_PER_GPU * SLURM_GPUS_ON_NODE * SLURM_JOB_NUM_NODES))
GPUS=$SLURM_GPUS_ON_NODE
CPUS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS_ON_NODE))

# additional environment variables for distributed training
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_PORT

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

if [[ -n $ckpt ]]
then
    argstring="$argstring --resume $ckpt"
fi

if [[ -n $desc ]]
then
    argstring="$argstring --desc $desc"
fi


python train.py $argstring

