#!/bin/bash

# README
# Phillip Long
# March 9, 2025

# Rename a model.

# bash /home/pnlong/jingyue_latents/rename_model.sh

# ARGUMENTS
##################################################

usage="Usage: $(basename ${0}) [-d] (data directory) [-r] (resume?) [-f] (fine tune?) [-sml] (small/medium/large) [-g] (gpu to use)"
while getopts ':d:g:rfsmlh' opt; do
  case "${opt}" in
    d)
      data_dir="${OPTARG}"
      ;;
    r)
      resume="--resume -1"
      ;;
    f)
      fine_tune="--fine_tune"
      ;;
    s)
      dim=512 # dimension
      layers=6 # layers
      heads=8 # attention heads
      ;;
    m)
      dim=768 # dimension
      layers=10 # layers
      heads=8 # attention heads
      ;;
    l)
      dim=960 # dimension
      layers=12 # layers
      heads=12 # attention heads
      ;;
    g)
      gpu="${OPTARG}"
      ;;
    h)
      echo ${usage}
      exit 0
      ;;
    :)
      echo -e "Option requires an argument.\n${usage}"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\n${usage}"
      exit 1
      ;;
  esac
done