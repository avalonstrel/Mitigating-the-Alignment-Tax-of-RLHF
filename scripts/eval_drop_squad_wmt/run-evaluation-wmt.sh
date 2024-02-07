# bash scripts/eval_drop_squad_wmt/run-mask-wmt.sh
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..
cd ../lmflow_benchmark
model_dir="${project_dir}/output_models"
log_dir="${project_dir}/eval_log"
dataset="lm_eval_wmt14"

model_tag="HuggingFaceH4/zephyr-7b-beta"
sed -i -e 's/\"use_cache\":\ false/\"use_cache\":\ true/g' ${model_dir}/${model_tag}/config.json
model_path=${model_dir}/${model_tag}
log_path=${log_dir}/${model_tag}/${dataset}
mkdir -p ${log_path}

bash scripts/run_benchmark_port.sh "${dataset}" ${model_path} 60020 0 \
    | tee -a ${log_path}/evaluation_final.log \
    2> ${log_path}/evaluation_final.err
