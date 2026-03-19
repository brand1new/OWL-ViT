#!/bin/bash

query_info=""
image_dir=""
output_digital_twins_dir=""
owlvit_checkpoint_path=""
depth_anything_checkpoint_path=""
sam_config_path=""
sam_checkpoint_path=""
out_log_path=""
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

echo "Start image to digital twins"
echo "    query_info: "$query_info
echo "    image_dir: "$image_dir
echo "    output_digital_twins_dir: "$output_digital_twins_dir
echo "    owlvit_checkpoint_path: "$owlvit_checkpoint_path
echo "    depth_anything_checkpoint_path: "$depth_anything_checkpoint_path
echo "    sam_config_path: "$sam_config_path
echo "    sam_checkpoint_path: "$sam_checkpoint_path
echo "    gpu_id:"$gpu_id
echo "    out_log_path: "$out_log_path

python ./digital_twins_pipeline/digital_twins_generation.py\
 --query_info $query_info\
 --image_dir $image_dir\
 --output_digital_twins_dir $output_digital_twins_dir\
 --owlvit_checkpoint_path $owlvit_checkpoint_path\
 --depth_anything_checkpoint_path $depth_anything_checkpoint_path\
 --sam_config_path $sam_config_path\
 --sam_checkpoint_path $sam_checkpoint_path\
 > $out_log_path

