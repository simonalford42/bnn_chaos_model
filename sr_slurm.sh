#!/usr/bin/env bash

 # job name
#SBATCH -J planet_sr
 # output file (%j expands to jobID)
#SBATCH -o out/sr_%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 32
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 02:00:00
#SBATCH --mem=50G
#SBATCH --partition=default_partition

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

python -u simon.py --total_steps 300000 --swa_steps 50000 --version 1278 --seed 1 --angles --no_mmr --no_nan --no_eplusminus "$@"