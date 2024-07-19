#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <subfolder> <logs_folder> <results_folder> <image_folder>"
    exit 1
fi

SUBFOLDER=$1
LOGS_FOLDER=$2
RESULTS_FOLDER=$3
IMAGE_FOLDER=$4

mkdir -p "$LOGS_FOLDER/$SUBFOLDER"
mkdir -p "$RESULTS_FOLDER/$SUBFOLDER"
source /opt/conda/etc/profile.d/conda.sh

file_path="$RESULTS_FOLDER/$SUBFOLDER/inv_ins.txt"
last_line=$(tail -n 1 "$file_path")
echo "$last_line"

conda activate visii
CUDA_VISIBLE_DEVICES=1 python train.py --image_folder "$IMAGE_FOLDER" --subfolder $SUBFOLDER --log_dir "$LOGS_FOLDER/$SUBFOLDER"

img_guidance_values=(1.5)
guidance_scale_values=(8 10 12)

for img_guidance in "${img_guidance_values[@]}"; do
    for guidance_scale in "${guidance_scale_values[@]}"; do
        echo "Running test.py with img_guidance=$img_guidance and guidance_scale=$guidance_scale"
        CUDA_VISIBLE_DEVICES=1 python test.py \
            --image_folder "$IMAGE_FOLDER/$SUBFOLDER" \
            --log_path "$LOGS_FOLDER/$SUBFOLDER" \
            --log_folder ip2p_${SUBFOLDER}_0_0.png \
            --guidance_scale $guidance_scale \
            --img_guidance $img_guidance --ct 1 --res_fol "$RESULTS_FOLDER"

        output_file="$RESULTS_FOLDER/$SUBFOLDER/only_ct_img_${img_guidance}_cond_${guidance_scale}.png"
#         echo "Saving output to $output_file"
        output_path="$RESULTS_FOLDER/$SUBFOLDER/best_only_ct_img_${img_guidance}_cond_${guidance_scale}.png"
        CUDA_VISIBLE_DEVICES=1 python get_output.py --fol1 "$IMAGE_FOLDER/$SUBFOLDER" --res_path "$RESULTS_FOLDER/$SUBFOLDER/ip2p_${SUBFOLDER}_0_0_1_0.png.png" --output_path $output_file
        CUDA_VISIBLE_DEVICES=1 python s_visual.py --fol1 "$IMAGE_FOLDER/$SUBFOLDER" --fol2 "$RESULTS_FOLDER/$SUBFOLDER" --res_file "$RESULTS_FOLDER/$SUBFOLDER/only_ct_img_${img_guidance}_cond_${guidance_scale}.json" --save_path $output_path --pref "$RESULTS_FOLDER/$SUBFOLDER/only_ct_img_${img_guidance}_cond_${guidance_scale}"

        CUDA_VISIBLE_DEVICES=1 python test.py \
            --image_folder "$IMAGE_FOLDER/$SUBFOLDER" \
            --log_path "$LOGS_FOLDER/$SUBFOLDER" \
            --log_folder ip2p_${SUBFOLDER}_0_0.png \
            --guidance_scale $guidance_scale \
            --img_guidance $img_guidance --ct 0 --hybrid_ins True --prompt "$last_line" --res_fol "$RESULTS_FOLDER"
        
        output_file="$RESULTS_FOLDER/$SUBFOLDER/only_llava_img_${img_guidance}_cond_${guidance_scale}.png"
#         echo "Saving output to $output_file"
        output_path="$RESULTS_FOLDER/$SUBFOLDER/best_only_llava_img_${img_guidance}_cond_${guidance_scale}.png"
        CUDA_VISIBLE_DEVICES=1 python get_output.py --fol1 "$IMAGE_FOLDER/$SUBFOLDER" --res_path "$RESULTS_FOLDER/$SUBFOLDER/ip2p_${SUBFOLDER}_0_0_1_0.png.png" --output_path $output_file
        CUDA_VISIBLE_DEVICES=1 python s_visual.py --fol1 "$IMAGE_FOLDER/$SUBFOLDER" --fol2 "$RESULTS_FOLDER/$SUBFOLDER" --res_file "$RESULTS_FOLDER/$SUBFOLDER/only_llava_img_${img_guidance}_cond_${guidance_scale}.json" --save_path $output_path --pref "$RESULTS_FOLDER/$SUBFOLDER/only_llava_img_${img_guidance}_cond_${guidance_scale}"

        CUDA_VISIBLE_DEVICES=1 python test.py \
            --image_folder "$IMAGE_FOLDER/$SUBFOLDER" \
            --log_path "$LOGS_FOLDER/$SUBFOLDER" \
            --log_folder ip2p_${SUBFOLDER}_0_0.png \
            --guidance_scale $guidance_scale \
            --img_guidance $img_guidance --ct 1 --hybrid_ins True --prompt "$last_line" --res_fol "$RESULTS_FOLDER"

        output_file="$RESULTS_FOLDER/$SUBFOLDER/ct_llava_img_${img_guidance}_cond_${guidance_scale}.png"
#         echo "Saving output to $output_file"
        output_path="$RESULTS_FOLDER/$SUBFOLDER/best_ct_llava_img_${img_guidance}_cond_${guidance_scale}.png"
        CUDA_VISIBLE_DEVICES=1 python get_output.py --fol1 "$IMAGE_FOLDER/$SUBFOLDER" --res_path "$RESULTS_FOLDER/$SUBFOLDER/ip2p_${SUBFOLDER}_0_0_1_0.png.png" --output_path $output_file
        CUDA_VISIBLE_DEVICES=1 python s_visual.py --fol1 "$IMAGE_FOLDER/$SUBFOLDER" --fol2 "$RESULTS_FOLDER/$SUBFOLDER" --res_file "$RESULTS_FOLDER/$SUBFOLDER/ct_llava_img_${img_guidance}_cond_${guidance_scale}.json" --save_path $output_path --pref "$RESULTS_FOLDER/$SUBFOLDER/ct_llava_img_${img_guidance}_cond_${guidance_scale}"
    done
done

conda deactivate
