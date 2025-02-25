#!/bin/bash

# README
# Phillip Long
# February 25, 2025

# Helper script to store command prompts for training models.

# bash /home/pnlong/jingyue_latents/train.sh > /home/pnlong/jingyue_latents/train.txt

# VARIABLES
##################################################

set -e

SOFTWARE_DIR="/home/pnlong/jingyue_latents"
BASE_DIR="/deepfreeze/user_shares/pnlong/jingyue_latents"
DEFAULT_GPU=0

# separator line functions
function major_separator_line {
    printf "=%.0s" {1..100}
    printf "\n"
}
function minor_separator_line {
    printf "* %.0s" {1..50}
    printf "\n"
}

##################################################


# EMOTION
##################################################

# helper variables
base_dir_name="emotion"
jingyue_data_dir="/deepfreeze/user_shares/jingyue/EMOPIA_emotion_recognition"
software_dir="${SOFTWARE_DIR}/${base_dir_name}"
base_dir="${BASE_DIR}/${base_dir_name}"
data_dir="${base_dir}/data"

# title
echo "${base_dir_name^^}"
major_separator_line

# create dataset
echo "Dataset:"
echo python ${software_dir}/dataset.py --data_dir "${jingyue_data_dir}/data_rvq_tokens_test" --partitions_dir "${jingyue_data_dir}/data_splits" --output_dir "${base_dir}"

minor_separator_line

# initial model
echo "Initial Model:"
echo python ${software_dir}/train.py --paths_train "${data_dir}/train.txt" --paths_train "${data_dir}/valid.txt" --output_dir "${base_dir}" --use_wandb --model_name "initial" --gpu "${DEFAULT_GPU}"

minor_separator_line

# evaluate
echo "Evaluate:"
echo python ${software_dir}/evaluate.py --paths_test "${data_dir}/test.txt" --models_list "${base_dir}/models.txt" --gpu "${DEFAULT_GPU}"

major_separator_line
echo

##################################################


# CHORD
##################################################

# helper variables
base_dir_name="chord"
jingyue_data_dir="/deepfreeze/user_shares/jingyue/chord_progression_detection"
software_dir="${SOFTWARE_DIR}/${base_dir_name}"
base_dir="${BASE_DIR}/${base_dir_name}"
data_dir="${base_dir}/data"

# title
echo "${base_dir_name^^}"
major_separator_line

# create dataset
echo "Dataset:"
echo python ${software_dir}/dataset.py --data_dir "${jingyue_data_dir}/data_rvq_tokens_test" --partitions_dir "${jingyue_data_dir}/data_splits" --output_dir "${base_dir}"

minor_separator_line

# initial model
echo "Initial Model:"
echo python ${software_dir}/train.py --paths_train "${data_dir}/train.txt" --paths_train "${data_dir}/valid.txt" --output_dir "${base_dir}" --use_wandb --model_name "initial" --gpu "${DEFAULT_GPU}"

minor_separator_line

# evaluate
echo "Evaluate:"
echo python ${software_dir}/evaluate.py --paths_test "${data_dir}/test.txt" --models_list "${base_dir}/models.txt" --gpu "${DEFAULT_GPU}"

major_separator_line
echo

##################################################


# STYLE
##################################################

# helper variables
base_dir_name="style"
jingyue_data_dir="/deepfreeze/user_shares/jingyue/style_classification"
software_dir="${SOFTWARE_DIR}/${base_dir_name}"
base_dir="${BASE_DIR}/${base_dir_name}"
data_dir="${base_dir}/data"

# title
echo "${base_dir_name^^}"
major_separator_line

# create dataset
echo "Dataset:"
echo python ${software_dir}/dataset.py --data_dir "${jingyue_data_dir}/data_rvq_tokens_test" --partitions_dir "${jingyue_data_dir}/data_splits" --output_dir "${base_dir}"

minor_separator_line

# initial model
echo "Initial Model:"
echo python ${software_dir}/train.py --paths_train "${data_dir}/train.txt" --paths_train "${data_dir}/valid.txt" --output_dir "${base_dir}" --use_wandb --model_name "initial" --gpu "${DEFAULT_GPU}"

minor_separator_line

# evaluate
echo "Evaluate:"
echo python ${software_dir}/evaluate.py --paths_test "${data_dir}/test.txt" --models_list "${base_dir}/models.txt" --gpu "${DEFAULT_GPU}"

major_separator_line
echo

##################################################
