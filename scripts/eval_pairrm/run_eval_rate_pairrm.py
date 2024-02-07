import os
import argparse
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, default="")
parser.add_argument("--gpu", type=int, default=-1, help="gpu id")
parser.add_argument("--save_path", default="alpaca_eval2_pairrm.csv", type=str, help="save path")
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import AutoTokenizer
from typing import List
import numpy as np
import torch
from tqdm.auto import tqdm, trange
import pandas as pd


pairrm = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf", device_map="cuda:7").eval()
tokenizer = AutoTokenizer.from_pretrained('llm-blender/PairRM-hf')
source_prefix = "<|source|>"
cand1_prefix = "<|candidate1|>"
cand2_prefix = "<|candidate2|>"


def tokenize_pair(sources: List[str], candidate1s: List[str], candidate2s: List[str], source_max_length=1224,
                  candidate_max_length=412):
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(cand1_prefix + candidate1s[i], max_length=candidate_max_length,
                                          truncation=True)
        candidate2_ids = tokenizer.encode(cand2_prefix + candidate2s[i], max_length=candidate_max_length,
                                          truncation=True)
        ids.append(source_ids + candidate1_ids + candidate2_ids)
    encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
    return encodings





def evaluate(args, sys_prompt_type, eval_data_type, ref_type):
    args.model_name = args.load_path.split("/")[-1].split(".json")[0]
    df_candidate = pd.read_json(args.load_path)
    assert eval_data_type in ["alpaca", "hh_rlhf"]
    if eval_data_type == "alpaca":
        if ref_type == 'gpt4':
            ref_path = "path/to/alpaca_eval_gpt4_baseline.json"        
    elif eval_data_type == "hh_rlhf":    
        ref_path = "path/to/rlhf/rlhf_eval_ref/pairrm_2k.json"
    # make sure the order of "instruction" is the same
    print(f"Reference Data: {ref_path}")
    df_reference = pd.read_json(ref_path)
    print(len(df_reference['instruction']), len(df_candidate['instruction']))
    print(len(df_reference['instruction'][0]), len(df_candidate['instruction'][0]))
    assert (df_reference['instruction'] == df_candidate['instruction']).all()
    prompts = df_reference['instruction'].values
    responses_A = df_candidate['output'].values
    responses_B = df_reference['output'].values
    batch_size = 16
    n_batches = len(prompts) // batch_size + 1
    all_batch_idxes = np.array_split(np.arange(len(prompts)), n_batches)
    with torch.no_grad():
        pairrm.eval()
        comparison_results = []
        for i in trange(n_batches,desc='batch',leave=False):
            batch_idxes = all_batch_idxes[i]
            encodings = tokenize_pair(prompts[batch_idxes], responses_A[batch_idxes], responses_B[batch_idxes])
            encodings = {k: v.to(pairrm.device) for k, v in encodings.items()}
            outputs = pairrm(**encodings)
            logits = outputs.logits.tolist()
            comparison_results.append(outputs.logits > 0)
    comparison_results = torch.cat(comparison_results).cpu().numpy()
    win_rate = comparison_results.mean()
    avg_length = np.mean([len(x) for x in responses_A])
    
    row = {"model": args.model_name, "win_rate": win_rate, 'avg_length': avg_length, 'reference': ref_type, "judge": "pairrm",}
    # append to the result file
    df_result = pd.DataFrame([row])

    df_result.to_csv(args.save_path, index=False, mode="a", header=not os.path.exists(args.save_path))

project_dir = "path/to/result"
sys_prompt_type, eval_data_type, ref_type = [1, "alpaca", "gpt4"]  #[1, "hh_rlhf", "hh_rlhf"]
keys = ["experiment_tags"]
for key in keys:
    load_paths = [
        os.path.join(project_dir, f"{key}/pairrm_v{sys_prompt_type}_{eval_data_type}.json")
    ]
    args.save_path = os.path.join(project_dir, f"{key}/pairrm_v{sys_prompt_type}_{eval_data_type}_{ref_type}_results.csv")
    for load_path in tqdm(load_paths, desc='file'):
        print('Load path:', load_path)
        args.load_path = load_path
        evaluate(args, sys_prompt_type, eval_data_type, ref_type)