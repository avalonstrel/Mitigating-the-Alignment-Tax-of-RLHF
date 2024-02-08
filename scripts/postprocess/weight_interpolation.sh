#!/bin/bash

gpu_idx=0
alpha=0.5
model_path0=openlm-research/open_llama_3b
model_path1=openlm-research/open_llama_3b
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..

weight_ensamble_names_paths="${model_path0} ${model_path1}"
weight_ensamble_save_path=output_models/test/ma_${alpha}_tag0_tag1
if [ ! -d "${weight_ensamble_save_path}" ];
then
  mkdir -p ${weight_ensamble_save_path}
fi

CUDA_VISIBLE_DEVICES=${gpu_idx} \
    python \
    scripts/postprocess/weight_interpolation.py \
    --model_name_or_path openlm-research/open_llama_3b \
    --weight_ensamble_names_paths ${weight_ensamble_names_paths} \
    --weight_ensamble_ratios ${alpha} \
    --torch_dtype bfloat16 \
    --weight_ensamble_save_path "${weight_ensamble_save_path}" \
    --dataset_path data \
    --deepspeed configs/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy 
