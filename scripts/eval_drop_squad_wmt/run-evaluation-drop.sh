# bash scripts/eval_drop_squad_wmt/run-mask-drop.sh
project_dir=$(cd "$(dirname $0)"/..; pwd)/../..
cd ../opencompass
model_dir="${project_dir}/output_models"
log_dir="${project_dir}/eval_log"
dataset="lm_eval_drop"

model_tag="HuggingFaceH4/zephyr-7b-beta"
sed -i -e 's/\"use_cache\":\ false/\"use_cache\":\ true/g' ${model_dir}/${model_tag}/config.json
model_path=${model_dir}/${model_tag}
log_path=${log_dir}/${model_tag}/${dataset}
mkdir -p ${log_path}

CUDA_VISIBLE_DEVICES=0 python run.py --datasets drop_gen \
                                    --hf-path ${model_path} \
                                    --model-kwargs device_map='auto' \
                                    --max-out-len 100 \
                                    --max-seq-len 2048 \
                                    --batch-size 6 \
                                    --num-gpus 1 \
| tee -a ${log_path}/evaluation_final.log \
2> ${log_path}/evaluation_final.err
