#!/bin/bash
# bash scripts/eval_raft/run_eval_raft_align.sh

gpu_ids="0,1,2,3"
port=11002

prefix="./scripts/eval_raft"
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..
base_dir="${project_dir}/path/to/save/generated_text"
mkdir -p $base_dir

test_model="${project_dir}/path/to/test_model"
reward_model="${project_dir}/path/to/reward_model" 

mkdir -p $base_dir/infer_set
mkdir -p $base_dir/filtered_set

bash ${prefix}/infer_get_samples.sh ${test_model} 0 ${base_dir}/infer_set ${gpu_ids} ${port}
bash ${prefix}/infer_get_rewards.sh ${base_dir}/infer_set ${base_dir}/filtered_set ${base_dir} ${reward_model} ${gpu_ids} ${port}
