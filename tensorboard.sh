#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus=1
#SBATCH --mem=16GB

PATH=$1
PORT=${2:-6006}

tensorboard --logdir $PATH --port $PORT