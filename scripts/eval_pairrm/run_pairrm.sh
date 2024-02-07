# bash scripts/eval_pairrm/run_text_generation.sh
model_path="path/to/model"
save_path="path/to/save"  #should be consistent with the path in run_eval_rate_pairrm.py
gpu="0"
mkdir -p ${save_path}
CUDA_VISIBLE_DEVICES=${gpu} python scripts/eval_pairrm/text_generation.py \
                        --model_path ${model_path} \
                        --gpu ${gpu} \
                        --save_path ${save_path}

python run_eval_rate_pairrm.py