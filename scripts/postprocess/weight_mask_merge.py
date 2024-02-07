#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser
import torch.nn.functional as F
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
import torch
import torch.nn as nn

pipeline_name = "evaluator"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

# Get the paths and ratios of weight-ensamble models.
weight_ensamble_names_paths = pipeline_args.weight_ensamble_names_paths
weight_ensamble_save_path = pipeline_args.weight_ensamble_save_path
alphas_path = pipeline_args.alphas_path

print('Model Paths:', weight_ensamble_names_paths)
print('Alphas Paths:', alphas_path)

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

# base_model = AutoModel.get_model(
#     model_args, 
#     tune_strategy='none', 
#     ds_config=ds_config, 
#     use_accelerator=pipeline_args.use_accelerator_for_evaluator
# )

# base_backend_model = base_model.get_backend_model()
# print('Finish load base model:', base_model)
# Load models.
# base_model = None
# backend_models = []
# merge_model_path = weight_ensamble_names_paths[1]
# merge_ckpt = torch.load(os.path.join(merge_model_path, 'pytorch_model.bin'))
# merge_method = 'linear'

# if 'graft' in merge_model_path:
#     merge_method = 'graft'
# print(merge_ckpt)

def load_model(model_path):
    model_args.model_name_or_path = model_path
    print('loading:', model_path)
    model = AutoModel.get_model(
        model_args, 
        tune_strategy='none', 
        ds_config=ds_config, 
        use_accelerator=pipeline_args.use_accelerator_for_evaluator
    )
    backend_model = model.get_backend_model().to('cpu')
    model = model
    print('Finish load base model:', model_path)
    return backend_model, model

## Load base model
base_backend_model, base_model = load_model(weight_ensamble_names_paths[0])

## Load ft model
ft_backend_model, ft_model = load_model(weight_ensamble_names_paths[1])

def update_by_wise_norm_sigmoid_linear(init_val, epsilon, w_type, key, mask_alphas, normalized_alphas,
                          base_state_dicts, ft_state_dicts, updated_state_dicts):
    """
    w_type: weight type ['weight','bias']
    key: linear model key
    """
    
    # selected_keys = ['model.layers.0.self_attn.k_proj', 'model.layers.15.self_attn.k_proj', 'model.layers.20.self_attn.k_proj']
    if (key + f'.{w_type}') not in base_state_dicts:
        return
    base_weight = base_state_dicts[key + f'.{w_type}']
    ft_weight = ft_state_dicts[key + f'.{w_type}']
    if 'lm_head' in key:
        wise_alpha = torch.tensor(init_val).to(device)
    else:
        mask_alpha = mask_alphas[key].to(device)
        wise_alpha = torch.sigmoid(mask_alpha) + epsilon
        normalized_alphas = [torch.sigmoid((normalized_alphas[k_]).to(device)) + epsilon for k_ in normalized_alphas ]
        wise_alpha = init_val * wise_alpha / sum(normalized_alphas) * len(normalized_alphas)
    print('Update w', key, wise_alpha, 'eps', epsilon)
    updated_weight = base_weight * wise_alpha + (1 - wise_alpha) * ft_weight
    updated_state_dicts[key + f'.{w_type}'] = updated_weight
    return updated_weight

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

updated_state_dicts = {}
base_state_dicts = base_backend_model.state_dict()
ft_state_dicts = ft_backend_model.state_dict()
device=torch.device('cpu')
merge_method = 'mask_norm_sigmoid_linear'
## @TODO Potential Extension
if 'mask_norm_sigmoid_linear' in alphas_path:
    merge_method = 'mask_norm_sigmoid_linear'

wise_data = torch.load(os.path.join(alphas_path, 'mask_alphas.bin'))
assert 'init_val' in wise_data, 'There should be a init val.'
init_val = wise_data['init_val']
mask_alphas = wise_data['mask_alphas']

base_gammas = {}
if 'base_gammas' in wise_data:
    base_gammas = wise_data['base_gammas']

epsilon = 1e-6
if 'epsilon' in wise_data:
    epsilon = wise_data['epsilon']

normalized_alphas = None
if 'mask_bkp_alphas' in wise_data:
    normalized_alphas = wise_data['mask_bkp_alphas']

print('Mask alphas:', mask_alphas)
print(f'Merge Method: {merge_method}, Init val:{init_val}, Epsilon:{epsilon}, Normalized alphas:{normalized_alphas}.')

key_list = [key for key, _ in base_backend_model.named_modules()]
for key in key_list:
    parent0, target0, target0_name = _get_submodules(base_backend_model, key)
    key_terms = key.split('.')
    if isinstance(target0, nn.Linear): 
        for w_type in ['weight', 'bias']:
            if merge_method == 'mask_norm_sigmoid_linear':
                if 'lm_head' in key:
                    if 'final0' in alphas_path:
                        init_val = 0
                update_by_wise_norm_sigmoid_linear(init_val, epsilon, w_type, key, mask_alphas,  normalized_alphas,
                            base_state_dicts, ft_state_dicts, updated_state_dicts)
            else:
                print(f'{key} No Merge.')

for key in base_state_dicts:
    if key not in updated_state_dicts:
        updated_state_dicts[key] = base_state_dicts[key]

base_backend_model.load_state_dict(updated_state_dicts)
print(weight_ensamble_save_path)
base_model.save(weight_ensamble_save_path)
