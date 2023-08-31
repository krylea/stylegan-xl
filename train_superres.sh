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
desc=$3
ckpt=${4:-''}
kimg=${5:-10000}
up_factor=${6:-2}
#MASTER_PORT=$7  # accept the master port as a user input

PREV_RES=$((RES / up_factor))

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

BATCH=$((BATCH_PER_GPU * SLURM_GPUS_ON_NODE)) 
GPUS=$SLURM_GPUS_ON_NODE
CPUS=$((SLURM_CPUS_PER_GPU * SLURM_GPUS_ON_NODE))

# additional environment variables for distributed training
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR="$(hostname -s)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

prev_run="training-runs/${DATASET_NAME}/${desc}_${PREV_RES}/best_model.pkl"

if [[ $DATASET_NAME == 'imagenet' ]]
then
    argstring="--outdir=./training-runs/$DATASET_NAME --cfg=stylegan3-t --data=/scratch/hdd001/datasets/imagenet/train --dataset_name $DATASET_NAME \
            --gpus=$SLURM_GPUS_ON_NODE --batch=$BATCH --mirror=1 --snap 30 \
            --batch-gpu $BATCH_PER_GPU --kimg $kimg --syn_layers 10 --workers $CPUS \
            --superres --up_factor $up_factor --head_layers 7 --restart_every 36000 --resolution $RES \
            --path_stem $prev_run"
else
    argstring=" --outdir=./training-runs/$DATASET_NAME --cfg=stylegan3-t --data=./data/${DATASET_NAME}${RES}.zip --dataset_name $DATASET_NAME \
            --gpus=$SLURM_GPUS_ON_NODE --batch=$BATCH --mirror=1 --snap 10 \
            --batch-gpu $BATCH_PER_GPU --kimg 10000 --syn_layers 7 --workers $CPUS \
            --superres --up_factor $up_factor --head_layers 4 --cbase 16384 --cmax 256 --restart_every 36000 --resolution $RES \
            --path_stem $prev_run"
fi

if [[ -n $ckpt ]]
then
    argstring="$argstring --resume $ckpt"
fi

if [[ -n $desc ]]
then
    argstring="$argstring --desc ${desc}_${RES}"
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

