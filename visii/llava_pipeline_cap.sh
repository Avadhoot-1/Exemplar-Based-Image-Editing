#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <subfolder> <logs_folder> <results_folder> <image_folder> <pt_folder>"
    exit 1
fi

SUBFOLDER=$1
LOGS_FOLDER=$2
RESULTS_FOLDER=$3
IMAGE_FOLDER=$4
PT_FOLDER=$5

mkdir -p "$LOGS_FOLDER/$SUBFOLDER"
mkdir -p "$RESULTS_FOLDER/$SUBFOLDER"
source /opt/conda/etc/profile.d/conda.sh

file_path="$RESULTS_FOLDER/$SUBFOLDER/inv_cap.txt"
last_line=$(tail -n 1 "$file_path")
echo "$last_line"

conda activate visii
img_guidance_values=(1.5)
guidance_scale_values=(8)

for img_guidance in "${img_guidance_values[@]}"; do
    for guidance_scale in "${guidance_scale_values[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python test_cap.py \
            --image_folder "$IMAGE_FOLDER/$SUBFOLDER" \
            --log_path "$LOGS_FOLDER/$SUBFOLDER" \
            --log_folder ip2p_${SUBFOLDER}_0_0.png \
            --guidance_scale $guidance_scale \
            --img_guidance $img_guidance --ct 1 --hybrid_ins True --prompt "$last_line" --res_fol "$RESULTS_FOLDER" --pt_fol "$PT_FOLDER"

    done
done

conda deactivate
