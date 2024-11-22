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
# python -u find_minima.py --total_steps 300000 --version $version --f1_variant bimt --f2_variant mlp

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

# first run of pysr on f2 (direct)
# python -u sr.py --version 29170 --target f2_direct # 29170 (linear)
# python -u sr.py --version 29170 --target f2 #4128301 (5489) (linear)
python -u sr.py --version 29170 --target f2 --time_in_hours 1 --seed 24

# second run of pysr on f2 (input is summary stats + equations)           
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/5489.pkl #  (linear, y=y-prod)
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/16506.pkl --time_in_hours 1  #  (linear, y=y)
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/37409.pkl --time_in_hours 1

# third run of pysr on f2
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/94758.pkl --previous_sr_path_2 sr_results/58805.pkl --seed 15

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

# 3rd run of pysr on f2 for the (OG):
# python -u sr.py --version 29170 --target f2 --sr_residual --previous_sr_path sr_results/92985.pkl # 5075721 (17290)
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/17290.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170 # 5077112

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

# for direct pysr validation loss evaluation: 4129853
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_model_selection best --total_steps 100 --load_f1 29170

# for residual (NN) pysr validation loss evaluation: 4129852 (linear, y=y-prod) |  (linear, y=y)
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_residual sr_results/4746.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_residual sr_results/18315.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170

# for residual (equations) pysr validation loss evaluation:
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5489.pkl --pysr_f2_residual sr_results/45730.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170
# 4896451    45730 (y=y) 
# 5074759    75373 (y=y-pred) 


##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

# python data_generation.py --n 10000 --output_path heron_data_2.pkl
# python heron.py --data_path heron_data_2.pkl --max_size 50 --time_in_hours 2 --seed 42
# python heron.py --data_path heron_data_2.pkl --time_in_hours 1 --sr_residual --previous_sr_path sr_results/59411.pkl --flag topk --selected_complexities 3 

# python data_generation.py --n 10000 --output_path heron_log_data.pkl
# python heron.py --data_path heron_log_data.pkl --max_size 50 --time_in_hours 24 --seed 42
# python heron.py --data_path heron_log_data.pkl --sr_residual --previous_sr_path sr_results/.pkl --flag all

# python mb_data_generation.py --n 10000 --output_path mb_data.pkl 
# python run_pysr_mb.py --data_path mb_data.pkl --include_mass --max_size 50 --time_in_hours 1
# python run_pysr_mb.py --data_path mb_data.pkl --include_mass --time_in_hours 1 --sr_residual --previous_sr_path sr_results_mb/40962.pkl --flag all


