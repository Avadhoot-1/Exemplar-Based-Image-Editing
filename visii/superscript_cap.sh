#!/bin/bash

# Check if the required arguments are provided
echo "$#"
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <logs_folder> <results_folder> <image_folder> <Pt_folder>"
    exit 1
fi

LOGS_FOLDER=$1
RESULTS_FOLDER=$2
IMAGE_FOLDER=$3
PT_FOLDER=$4

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
            ./llava_pipeline_cap.sh "$subdir_name" "$LOGS_FOLDER" "$RESULTS_FOLDER" "$IMAGE_FOLDER" "$PT_FOLDER"
        else
            echo "Skipping $subdir_name as it does not contain an underscore."
        fi
    fi
done
