#!/bin/bash
# bash scripts/galactica1.3B/postprocess/weight_ensamble.sh 4 11011 0.4 gala1.3b_finetune_pubmedqa_ep20
# /home/linhangyu/Projects/LLMs/LMFlow/output_models/gala1.3b_finetune_medmcqa_ep10/checkpoint-400

gpu_idx=0
alpha=0.5
model_path0=path/to/model/before/rlhf
model_path1=path/to/model/after/rlhf
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..

weight_ensamble_names_paths="${model_path0} ${model_path1}"
weight_ensamble_save_path=path/to/save/ma_${alpha}_tag0_tag1
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
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy 
