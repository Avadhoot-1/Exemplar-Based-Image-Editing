#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <logs_folder> <results_folder> <image_folder>"
    exit 1
fi

LOGS_FOLDER=$1
RESULTS_FOLDER=$2
IMAGE_FOLDER=$3

# Check if the provided image folder exists
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "The provided image folder does not exist."
    exit 1
fi

# Iterate over each subdirectory in the images directory
for subdir in "$IMAGE_FOLDER"/*/; do
    # Check if it is a directory
    if [ -d "$subdir" ]; then
        # Extract the subdirectory name using basename
        subdir_name=$(basename "$subdir")
        echo "Processing $subdir_name"

        # Check if subdir_name contains an underscore and is greater than 'a'
        if [[ "$subdir_name" == *_* ]]; then
            ./llava_pipeline.sh "$subdir_name" "$LOGS_FOLDER" "$RESULTS_FOLDER" "$IMAGE_FOLDER"
        else
            echo "Skipping $subdir_name as it does not contain an underscore."
        fi
    fi
done
