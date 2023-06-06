#!/bin/bash
#SBATCH --job-name=stylegan-xl
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB

src=$1
tgt=$2
res=$3
args=${@:4}

python dataset_tool.py --source=$src --dest=$tgt --resolution="${res}x${res}" --transform=center-crop $args

