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
#SBATCH -t 24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

# gets the next available version number
version=$(python versions.py)

# now apply l2 reg to f2, freezing f1
python -u find_minima.py --version $version --total_steps 150000 --f2_variant linear --l1_reg f2_weights --l1_coeff 2 --load_f1_f2 24880 --freeze_f1 "$@"
version2=$(python versions.py)
# prune topk from f2, then continue training for a bit
python -u find_minima.py --version $version2 --total_steps 150000 --load_f1_f2 $version --prune_f2_topk 2 --freeze_f1 "$@"

