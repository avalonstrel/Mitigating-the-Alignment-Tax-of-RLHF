# bash scripts/evaluation_cs_qa/run_evaluation.sh

project_dir=$(cd "$(dirname $0)"/..; pwd)/../..
cd ${project_dir}/../lm-evaluation-harness
log_dir=${project_dir}/path/to/save
if [ ! -d "${log_dir}" ];
then 
    mkdir -p ${log_dir}
fi
model_path=path/to/model
eval_log_path=${log_dir}/result.json
gpu_idx=0
batch_size=16
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="\"${model_path}\"" \
    --tasks arc_easy,arc_challenge,race,boolq,piqa \
    --output_path ${eval_log_path} \
    --batch_size ${batch_size} \
    --max_batch_size ${batch_size} \
    --no_cache \
    --device cuda:${gpu_idx} \
    | tee ${log_dir}/evaluation.log \
    2> ${log_dir}/evaluation.err
