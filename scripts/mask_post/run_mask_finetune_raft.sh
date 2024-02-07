#!/bin/bash


# bash scripts/mask_post/run_mask_finetune_raft.sh
# hyper-parameters
approach=mask_norm_sigmoid_linear
mask_level=layerwise
lr=2e-5
warp_init_val=0.2
reg_alpha=1e-4
sum_reg_type=0.2
epsilon=0.2
gpu_ids="0,1"

port=29500

exp_id=${approach}_${mask_level}_ft_raft_${lr}_${warp_init_val}_reg${reg_alpha}_sr${sum_reg_type}_eps${epsilon}_ep1
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..
log_dir=${project_dir}/log/${exp_id}
model_path=${project_dir}/path/to/after/rlhf
output_dir=${model_path}/mask_opt/${exp_id}
dataset_path=${project_dir}/path/to/data/collected

if [ ! -d "${output_dir}" ];
then
  mkdir -p ${output_dir}
fi

if [ ! -d "${log_dir}" ];
then
  mkdir -p ${log_dir}
fi

accelerate launch --main_process_port ${port} --gpu_ids=${gpu_ids} \
  examples/finetune_mask.py \
    --model_name_or_path ${model_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate ${lr}\
    --block_size 512 \
    --per_device_train_batch_size 2 \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --warp_init_val ${warp_init_val} \
    --approach ${approach} \
    --reg_alpha ${reg_alpha} \
    --sum_reg_type ${sum_reg_type} \
    --epsilon ${epsilon} \
    --mask_level ${mask_level} \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 1000 \
    --dataloader_num_workers 4 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
