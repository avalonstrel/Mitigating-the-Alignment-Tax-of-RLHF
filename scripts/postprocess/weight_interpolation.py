import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

pipeline_name = "evaluator"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

# Get the paths and ratios of weight-ensamble models.
weight_ensamble_names_paths = pipeline_args.weight_ensamble_names_paths
weight_ensamble_ratios = pipeline_args.weight_ensamble_ratios
weight_ensamble_save_path = pipeline_args.weight_ensamble_save_path
weight_ensamble_ratios.append(1 - weight_ensamble_ratios[0])
assert len(weight_ensamble_ratios) == 2, 'Only 2 merge is supported.'
print('Model Paths:', weight_ensamble_names_paths)
print('Model Ratio:', weight_ensamble_ratios)
with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

# Load models.
base_model = None
backend_models = []
for model_path in weight_ensamble_names_paths:
    model_args.model_name_or_path = model_path
    print('loading:', model_path)
    model = AutoModel.get_model(
        model_args, 
        tune_strategy='none', 
        ds_config=ds_config, 
        use_accelerator=pipeline_args.use_accelerator_for_evaluator
    )
    model.get_backend_model().eval()
    backend_models.append(model.get_backend_model().to('cpu'))
    if base_model is None:
        base_model = model
    print('Finish load:', model_path)
base_backend_model = backend_models[0]
print('Finish load All:', base_backend_model)

merge_method = weight_ensamble_save_path.split('_')[-2]
print(f'Merge Method:{merge_method}.')

def merge_split(merge_method, key, weights, ori_ratio):
    merge_terms = merge_method.split('|')  #split|6,13|0.2|0.5|0.2
    split_layers = merge_terms[1].split('#')
    split_layers = [int(split_layer) for split_layer in split_layers]
    
    assert len(split_layers) == len(merge_terms) - 3
    ratios = [float(t) for t in merge_terms[2:]]
    
    terms = key.split('.')
    layer_idx, ratio = 0, ratios[0]

    if 'split0' in merge_method and 'lm_head' in key:
        ratio = 0
    elif 'lm_head' in key or 'norm' in key:
        ratio = ori_ratio
        
    if terms[0] == 'model' and terms[1] == 'layers':
        layer_idx = int(terms[2])
        ratio = ratios[0]
        for s_i, split_layer in enumerate(split_layers):
            if layer_idx > split_layer:
                ratio = ratios[s_i + 1]
    print(key, layer_idx, ratio)
    return weights[0] * ratio + weights[1] * (1 - ratio)

def merge_direct(weights, ratio):
    return weights[0] * ratio + weights[1] * (1 - ratio)

updated_state_dict = {}
for key in base_backend_model.state_dict():
    weights = [backend_model.state_dict()[key] for backend_model in backend_models]
    if 'split' in merge_method:
        updated_weight = merge_split(merge_method, key, weights, weight_ensamble_ratios[0])
    else:
        updated_weight = merge_direct(weights, weight_ensamble_ratios[0])
    updated_state_dict[key] = updated_weight
    
base_backend_model.load_state_dict(updated_state_dict)
print(weight_ensamble_save_path)
base_model.save(weight_ensamble_save_path)
