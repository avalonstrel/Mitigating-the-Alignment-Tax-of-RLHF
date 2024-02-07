from datasets import load_dataset
import torch
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
    AutoTokenizer
)
import sys

import numpy as np
import matplotlib.pyplot as plt
import random
import time

import re



data_files = [
    #"/home/xiongwei/rm_study/LMFlow/data/open_llama_7b_replay/10/raft_19iter.json",
    # "/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/replay_exp/data/pretrained/sub_sampled/togethercomputer_train_100M_v3.jsonl",
    # "/home/jianmeng/forgetting_data/v4/togethercomputer_train_200M_2048-4096.jsonl",
    #  "/home/jianmeng/forgetting_data/v4/togethercomputer_train_200M_4096-9192.jsonl"],
    # "/home/xiongwei/over_opt/LMFlow_RAFT_Dev/output_models/replay_exp/data/raft_3b/raft_data_16M_tokens.json"
    # "/home/linhangyu/Projects/LLMs/LMFlow_RAFT_Dev/data/mixture_exp_3b/10/raft_data_26624.json"
    "/home/jianmeng/linhangyu/Projects/LLMs/LMFlow_RAFT_Dev/data/10w_sharegpt/sharegpt_en_10w.json"
]

all_texts = []
raft = load_dataset("json", data_files=data_files[0], split="train", field="instances")


tokenizer = AutoTokenizer.from_pretrained("/home/jianmeng/linhangyu/Projects/LLMs/LMFlow_RAFT_Dev/output_models/sft_open_llama_3b_1epoch_plus_hh_rlhf_1epoch")


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["text"])
    sample["query"] = sample['text'] #tokenizer.decode(sample["input_ids"])
    return sample



raft = raft.map(tokenize, batched=False)

print(len(raft))
all_raft_tokens = np.sum([len(sample['input_ids']) for sample in raft])
print(all_raft_tokens)

import random
import json
all_texts = []
all_texts.extend(raft['text'])
random.shuffle(all_texts)
all_texts = all_texts[:(len(all_texts)//2)]
store_texts = [{"text":txt} for txt in all_texts]

output_dataset = {}
output_dataset['type'] = "text_only"
output_dataset['instances'] = store_texts
with open("/home/jianmeng/linhangyu/Projects/LLMs/LMFlow_RAFT_Dev/data/5w_sharegpt/5w_sharegpt.json", 'w', encoding='utf8') as f:
    json.dump(output_dataset, f, ensure_ascii=False)