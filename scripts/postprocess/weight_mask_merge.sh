#!/bin/bash
gpu_idx=0
master_port=11000
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..
model_path0=${project_dir}/path/to/before/rlhf
model_path1=${project_dir}/path/to/after/rlhf

alphas_path=${project_dir}/path/to/mask_alpha.bin
weight_ensamble_names_paths="${model_path0} ${model_path1}"
weight_ensamble_save_path=${project_dir}/path/to/save

if [ ! -d "${weight_ensamble_save_path}" ];
then
  mkdir -p ${weight_ensamble_save_path}
fi

deepspeed_args="--master_port=${master_port} --include localhost:${gpu_idx}"
deepspeed ${deepspeed_args} \
    scripts/llama3b/postprocess/weight_mask_merge.py \
    --model_name_or_path openlm-research/open_llama_3b \
    --alphas_path ${alphas_path} \
    --weight_ensamble_names_paths ${weight_ensamble_names_paths} \
    --weight_ensamble_ratios 0.0 \
    --weight_ensamble_save_path "${weight_ensamble_save_path}" \
    --dataset_path data \
    --deepspeed configs/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy 