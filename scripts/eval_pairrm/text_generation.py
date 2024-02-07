# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate
from collections.abc import Iterable
import os
import torch
from transformers import pipeline
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir="/home/jianmeng/linhangyu/Projects/LLMs/LMFlow_RAFT_Dev"
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="zephr-7b-beta")
parser.add_argument("--eval_data_type", type=str, default="alpaca")
parser.add_argument("--sys_prompt_type", type=int, default=1)
parser.add_argument("--gpu", type=int, default=-1, help="gpu id")
parser.add_argument("--save_path", default="alpaca_eval2_pairrm.csv", type=str, help="save path")
args = parser.parse_args()

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(args.model_path,  torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model.to(device)
sys_prompt_dict = {
    0:"You are a friendly chatbot who always responds in the style of a pirate",
    1:"You are a helpful, honest and respectful chatbot.",
    2:"You are a friendly chatbot who always responds helpfully, honestly and respectfully.",
}

class HHRLHFData:
    def __init__(self, datasets, batch_size=50, sys_prompt_type=1) -> None:
        self.datasets = datasets
        self.batch_size = batch_size
        batch_num = len(self.datasets) // self.batch_size 
        if len(self.datasets) % self.batch_size != 0:
            batch_num += 1
        self.batch_num = batch_num
        self.sys_prompt = sys_prompt_dict[sys_prompt_type]

    def __len__(self):
        return self.batch_num
    
    def get_data(self, index):
        tmp_prompts = []
        tmp_inputs = []
        # print(index, self.batch_size)
        start_idx, end_idx = index * self.batch_size, (index + 1) * self.batch_size
        for sample in self.datasets[start_idx:end_idx]:
            input_content = sample["text"].replace("###Human: ", "").replace("###Assistant:", "")
            messages = [
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {"role": "user", "content": input_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tmp_prompts.append(prompt), tmp_inputs.append(input_content)
        return tmp_prompts, tmp_inputs
        
class AlpacaData:
    def __init__(self, datasets, batch_size=50, sys_prompt_type=0) -> None:
        self.datasets = datasets
        self.batch_size = batch_size
        batch_num = len(self.datasets) // self.batch_size 
        if len(self.datasets) % self.batch_size != 0:
            batch_num += 1
        self.batch_num = batch_num
        self.sys_prompt = sys_prompt_dict[sys_prompt_type]

    def __len__(self):
        return self.batch_num
    
    def get_data(self, index):
        tmp_prompts = []
        tmp_inputs = []
        # print(index, self.batch_size)
        start_idx, end_idx = index * self.batch_size, (index + 1) * self.batch_size
        for sample in self.datasets[start_idx:end_idx]:
            input_content = sample["instruction"]
            messages = [
                {
                    "role": "system",
                    # "content": "You are a friendly chatbot who always responds in the style of a pirate",
                    "content": self.sys_prompt,
                },
                {"role": "user", "content": input_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tmp_prompts.append(prompt), tmp_inputs.append(input_content)
        return tmp_prompts, tmp_inputs

# alpaca_params = {'temperature':0.7, 'top_p':1.0, 'max_new_tokens':300}
# hhrlhf_params = {'temperature':0.7, 'top_k':50, 'top_p':0.95, 'max_new_tokens':256}
sys_prompt_type = args.sys_prompt_type
eval_data_type = args.eval_data_type
if eval_data_type == "hh_rlhf":
    eval_datasets = json.load(open(f"{project_dir}/data/hh_rlhf/rlhf/rlhf_eval/eval_prompt_first_half.json", "r"))
    # Only Split 2000
    eval_datasets = eval_datasets["instances"][:2000]
    eval_datasets = HHRLHFData(eval_datasets, batch_size=40, sys_prompt_type=sys_prompt_type)
    sample_params = {'temperature':0.7, 'top_k':50, 'top_p':0.95, 'max_new_tokens':256}
elif eval_data_type == "alpaca":
    eval_datasets = json.load(open(f"{project_dir}/data/alpaca/alpaca_eval_gpt4_baseline.json", "r"))
    eval_datasets = AlpacaData(eval_datasets, batch_size=40, sys_prompt_type=sys_prompt_type)
    sample_params = {'temperature':0.7, 'top_p':1.0, 'max_new_tokens':300}
# print(sum([len(eval_datasets.get_data(i)) for i in range(len(eval_datasets))]))


output_datasets = []
for i in tqdm(range(len(eval_datasets))):
    sample, inputs = eval_datasets.get_data(i)
    model_inputs = tokenizer(sample, return_tensors="pt", padding=True).to(device)
    model_inputs = model_inputs.to(device)
    # model.to(device)
    generated_ids = model.generate(**model_inputs, do_sample=True, **sample_params)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    output_texts = [generated_text.split("\n<|assistant|>\n")[-1] for generated_text in generated_texts]

    for k in range(len(inputs)):
        output_datasets.append(
            {
                "dataset":eval_data_type,
                "instruction": inputs[k],
                "output":output_texts[k],
                "generator":args.model_path
            }
        )
    # print(generated_texts[-1], output_datasets[-1])
    json.dump(output_datasets, open(os.path.join(args.save_path, f"pairrm_v{sys_prompt_type}_{eval_data_type}.json"), "w"))


# # pipe = pipeline("text-generation", model="HuggingFaceH4/mistral-7b-sft-beta", torch_dtype=torch.bfloat16, device_map=args.gpu)
# pipe = pipeline("text-generation", model=args.model_path, torch_dtype=torch.bfloat16, device_map=args.gpu, use_cache=True)
# #     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# #     if s_i % 10 == 0:
# #         print(f'Finish {s_i}/{len(eval_datasets)}.')
# #     # print(outputs[0]["generated_text"])
# #     output_content = {
# #         "dataset":"hh_rlhf",
# #         "instruction": input_content,
# #         "output":outputs[0]["generated_text"],
# #         "generator":args.model_path
# #     }
# #     output_datasets.append(output_content)



# def input_data():
#     for s_i, sample in enumerate(eval_datasets):
#         input_content = sample["text"].replace("###Human: ", "").replace("###Assistant:", "")
#         yield input_content
 

# # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# output_datasets = []
# ###################################################################################################
# # for s_i, sample in tqdm(enumerate(eval_datasets)):
# #     input_content = sample["text"].replace("###Human: ", "").replace("###Assistant", "")
# #     messages = [
# #         {
# #             "role": "system",
# #             "content": "You are a friendly chatbot who always responds in the style of a pirate",
# #         },
# #         {"role": "user", "content": input_content},
# #     ]
# #     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# #     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# #     if s_i % 10 == 0:
# #         print(f'Finish {s_i}/{len(eval_datasets)}.')
# #     # print(outputs[0]["generated_text"])
# #     output_content = {
# #         "dataset":"hh_rlhf",
# #         "instruction": input_content,
# #         "output":outputs[0]["generated_text"],
# #         "generator":args.model_path
# #     }
# #     output_datasets.append(output_content)
# ###################################################################################################
# for (input_content, outputs) in tqdm(zip(input_data(), 
#                                          pipe(data(), max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95))):

#     # generated_content = outputs[0]["generated_text"]
    
#     output_content = {
#         "dataset":"hh_rlhf",
#         "instruction": input_content,
#         "output":outputs[0]["generated_text"].split("\n<|assistant|>\n")[-1],
#         "generator":args.model_path
#     }
#     # print(output_content)
#     output_datasets.append(output_content)
    

# # <|system|>
# # You are a friendly chatbot who always responds in the style of a pirate.</s>
# # <|user|>
# # How many helicopters can a human eat in one sitting?</s>
# # <|assistant|>
# # Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!
