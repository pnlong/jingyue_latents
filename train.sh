#!/bin/bash

# README
# Phillip Long
# February 25, 2025

# Helper script to store command prompts for training models.

# VARIABLES
##################################################

BASE_DIR="/deepfreeze/user_shares/pnlong/jingyue_latents"
SOFTWARE_DIR="/home/pnlong/jingyue_latents"

##################################################


# 
base_dir="/emotion/"
data_dir="${base_dir}/data"

python ${software_dir}/emotion/train.py --paths_train "${data_dir}/train.txt" --paths_train "${data_dir}/valid.txt" --output_dir "${base_dir}" --model_name