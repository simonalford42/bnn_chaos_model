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

# source /home/sca63/mambaforge/etc/profile.d/conda.sh
# conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

version=$((1 + RANDOM % 999999))
version2=$((1 + RANDOM % 999999))

# python -u find_minima.py --total_steps 300000 --version $version --f1_variant linear --f2_variant mlp
# python -u find_minima.py --total_steps 300000 --version $version --f1_variant products3 --f2_variant mlp

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

# first run of pysr on f2 (direct)
# python -u sr.py --version 29170 --target f2_direct --seed 0 # 29170 (linear)
# python -u sr.py --version 8218 --target f2_direct --seed 0 # 8218 (products3)
# python -u sr.py --version 29170 --target f2 --seed 0 #job 4128301 (5489) (linear)
# python -u sr.py --version 8218 --target f2 --seed 0 #job 4128302 (21791) (products3) 
# python -u sr.py --version 43139 --target f2 --seed 0 # 43139 is Simon's linear (4129698) --> didn't work because 43139 only has mean, no std

# second run of pysr on f2 (input is summary stats + equations)           
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/5489.pkl --seed 0 # job 4129727 (4746) (linear, y=y-prod)
# python -u sr.py --version 8218 --target f2 --sr_residual --previous_sr_path sr_results/21791.pkl --seed 0 # job 4129755 (11946) (products3, y=y-prod)
python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/5489.pkl --seed 0 # job 4129762 (18315) (4158097) (linear, y=y)
# python -u sr.py --version 8218 --target f2 --sr_residual --previous_sr_path sr_results/21791.pkl --seed 0 # job 4129764 (75407) (products3, y=y)

# third run of pysr on f2
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/4746.pkl --seed 0 # job 4129759 (57868) (linear, y=y-prod)
# python -u sr.py --version 8218 --target f2 --sr_residual --previous_sr_path sr_results/.pkl --seed 0 # job  () (products3, y=y-prod)

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

# for direct pysr validation loss evaluation (OG): 4129836
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_model_selection best --total_steps 100 --load_f1 29170

# for residual (NN) pysr validation loss evaluation (OG): 4129851
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/92071.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170

# for residual (equations) pysr validation loss evaluation (OG): 4129835
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/92985.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/52420.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

# for direct pysr validation loss evaluation: 4129853
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_model_selection best --total_steps 100 --load_f1 29170

# for residual (NN) pysr validation loss evaluation: 4129852 (linear, y=y-prod) |  (linear, y=y)
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_residual sr_results/4746.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_residual sr_results/18315.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170

# for residual (equations) pysr validation loss evaluation: (linear, y=y-prod) | (linear, y=y)
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_residual sr_results/.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170
