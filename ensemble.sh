#!/usr/bin/env bash

 # job name
#SBATCH -J bnn_chaos
 # output file (%j expands to jobID)
#SBATCH -o out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)

#SBATCH -t 48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

version=$((1 + RANDOM % 999999))

for seed in `seq 0 2`; do
    python find_minima.py --version $version --seed $seed --slurm_id $SLURM_JOB_ID --slurm_name $SLURM_JOB_NAME
    python run_swag.py --version $version --seed $seed --slurm_id $SLURM_JOB_ID --slurm_name $SLURM_JOB_NAME
done
