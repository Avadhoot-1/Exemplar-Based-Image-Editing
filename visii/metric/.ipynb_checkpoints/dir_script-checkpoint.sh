#!/bin/bash

# Directory containing the images subdirectories
images_dir="/mnt/localssd/avadhoot/image_folders1"

# Check if images directory exists
if [ ! -d "$images_dir" ]; then
    echo "The directory $images_dir does not exist."
    exit 1
fi

# Iterate over each subdirectory in the images directory
for subdir in "$images_dir"/*/; do
    # Check if it is a directory
    if [ -d "$subdir" ]; then
        # Extract the subdirectory name using basename
        subdir_name=$(basename "$subdir")
        echo "Processing $subdir_name"

        # Check if subdir_name contains an underscore and is greater than 'a'
        if [[ "$subdir_name" == *_* ]]; then
            # Check if ./results/subdir_name exists
            CUDA_VISIBLE_DEVICES=1 python3 directional_clip.py --img_fol /mnt/localssd/avadhoot/image_folders1/$subdir_name --res_fol /mnt/localssd/avadhoot/results/$subdir_name
        else
            echo "Skipping $subdir_name as it does not contain an underscore or is not lexicographically greater than 'a'."
        fi
    fi
done

