#!/bin/bash

# README
# Phillip Long
# March 9, 2025

# Rename a model.

# bash /home/pnlong/jingyue_latents/rename_model.sh

# ARGUMENTS
##################################################

# stop if error
set -e

# usage
model_dir_argument_name="[model_dir]"
new_name_argument_name="[new_name]"
usage() {
  echo "Usage: $(basename ${0}) ${model_dir_argument_name} ${new_name_argument_name}"
  echo "  * ${model_dir_argument_name}: Absolute filepath to the directory of the relevant model, where \`basename ${model_dir_argument_name}\` yields the model's current name."
  echo "  * ${new_name_argument_name}:  New name for the relevant model."
  exit 1
}

# print help statement if flag specified
while getopts ":h" opt; do
  case "${opt}" in
    h)
      usage
      ;;
  esac
done

# ensure two positional arguments are provided
arguments_help_statement() {
  echo "Please provide two positional arguments, ${model_dir_argument_name} and ${new_name_argument_name}."
}
arguments_help_statement=""
for i in $(seq 1 2); do
  if [ -z "${!i}" ]; then
    arguments_help_statement
    usage
  fi
done
if [ ${#} -gt 2 ]; then
  arguments_help_statement
  usage
fi

# parse arguments
model_dir=${1}
new_name=${2}

# ensure model directory exists
if [ ! -d ${model_dir} ]; then
  echo "Invalid ${model_dir_argument_name} argument. \`${model_dir}\` does not exist."
  usage
fi

# scrape information from arguments
old_name=$(basename ${model_dir})
task_dir=$(dirname ${model_dir})

##################################################


# RENAME MODEL
##################################################

# replace in training files
sed -i "s+${old_name}+${new_name}+g" "${model_dir}/train_args.json" # arguments file
sed -i "s+${old_name}+${new_name}+g" "${model_dir}/train.log" # training log file
sed -i "s+${old_name}+${new_name}+g" "${task_dir}/models.txt" # models file

# replace in evaluation files if they exist
evaluate_log_filepath="${task_dir}/evaluate.log"
if [ -f ${evaluate_log_filepath} ]; then
  sed -i "s+${old_name}+${new_name}+g" ${evaluate_log_filepath} # evaluation log file
  sed -i "s+${old_name}+${new_name}+g" "${task_dir}/evaluation.loss.csv" # evaluation log file
  sed -i "s+${old_name}+${new_name}+g" "${task_dir}/evaluation.accuracy.csv" # evaluation log file
fi

# rename directory
mv "${model_dir}" "${task_dir}/${new_name}"

# remind me to rename model on WANDB
echo "Model renamed. Make sure to rename the model on WANDB."

##################################################