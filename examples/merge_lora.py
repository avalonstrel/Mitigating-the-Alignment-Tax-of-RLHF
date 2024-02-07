#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Merge base model and lora model into a full model.
"""

import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
import json
from lmflow.args import (
    ModelArguments,
    AutoArguments,
)

from lmflow.models.auto_model import AutoModel


@dataclass
class MergeLoraArguments:
    output_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "output merged full model path"
        },
    )


def main():
    pipeline_name = "evaluator"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, MergeLoraArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, merge_lora_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, merge_lora_args, pipeline_args = parser.parse_args_into_dataclasses()

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model_args.use_lora = True
    model = AutoModel.get_model(model_args,
                                tune_strategy='none',
                                ds_config=ds_config)
    loraA_key = "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"
    loraB_key = "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight"
    key ="base_model.model.model.layers.3.self_attn.q_proj.weight"
    # print(loraA_key, model.get_backend_model().state_dict()[loraA_key])
    # print(loraB_key, model.get_backend_model().state_dict()[loraB_key])
    # print(key, model.get_backend_model().state_dict()[key])
    model.merge_lora_weights()
    # after_key = "model.layers.3.self_attn.q_proj.weight"
    # print(list(model.get_backend_model().state_dict()))
    # print(after_key, model.backend_model_full.state_dict()[after_key])
    # print(after_key, model.get_backend_model().state_dict()[key])
    model.save(merge_lora_args.output_model_path, save_full_model=True)


if __name__ == '__main__':
    main()
