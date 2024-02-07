
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    Trainer
)

from transformers.trainer import *

if is_apex_available():
    from apex import amp

import deepspeed


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class MaskNormalizedSigmoidLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_wise_model(self, device, dtype, 
                       target0, target1, 
                       wise_weight_alpha, wise_bias_alpha, normalized_alphas,
                       init_val=0.2, epsilon=1e-6, alpha_type='single',
                       ):
        self.weight = nn.Parameter((target0.weight).to(device, dtype=dtype))
        self.init_val = init_val
        self.normalized_alphas = normalized_alphas
        self.epsilon = epsilon
        
        if wise_weight_alpha is not None:
            self.wise_weight_alpha = wise_weight_alpha.to(device=device, dtype=dtype)
        else:
            if alpha_type == 'single':
                self.wise_weight_alpha = nn.Parameter(torch.logit(torch.tensor(init_val)).to(device, dtype=dtype))
            elif alpha_type == 'full':
                self.wise_weight_alpha = nn.Parameter(torch.logit(torch.zeros_like(self.weight) + init_val).to(device, dtype=dtype))
        
        self.wise_weight = (target1.weight).to(device, dtype=dtype).detach()
        self.weight.requires_grad = False
        
        
        if self.bias is not None:
            self.bias = nn.Parameter((target0.bias).to(device, dtype=dtype))
            if wise_bias_alpha is not None:
                self.wise_bias_alpha = wise_bias_alpha.to(device=device, dtype=dtype)
            else:
                if alpha_type == 'single':
                    self.wise_bias_alpha = nn.Parameter(torch.logit(torch.tensor(init_val)).to(device, dtype=dtype))
                else:
                    self.wise_bias_alpha = nn.Parameter(torch.logit(torch.zeros_like(self.bias) + init_val).to(device, dtype=dtype))
            self.wise_bias = (target1.bias).to(device, dtype=dtype).detach()
            self.bias.requires_grad = False

    def forward(self, input):
        epsilon = self.epsilon
        normalized_alpha = self.init_val * (F.sigmoid(self.wise_weight_alpha) + epsilon) / sum([F.sigmoid(alpha) + epsilon for alpha in self.normalized_alphas]) * len(self.normalized_alphas)
        if self.bias is None:
            return F.linear(input, 
                            self.weight * normalized_alpha + (1 - normalized_alpha) * self.wise_weight,
                            self.bias)
        else:
            normalized_bias_alpha = self.init_val * (F.sigmoid(self.wise_bias_alpha) + epsilon) / sum([F.sigmoid(alpha) + epsilon for alpha in self.normalized_alphas]) * len(self.normalized_alphas)
            return F.linear(input, 
                            self.weight * normalized_alpha + (1 - normalized_alpha) * self.wise_weight,
                            self.bias * normalized_bias_alpha + (1 - normalized_bias_alpha) * self.wise_bias)


def create_alpha(method, init_val):
    if method == 'mask_norm_sigmoid_linear':
        return nn.Parameter(torch.logit(torch.tensor(init_val * 1.0)))

def get_max_layers_num(key_list):
    layer_idxs = [int(key.split('.')[2]) for key in key_list if 'model.layers' in key and len(key.split('.')) > 2]
    return max(layer_idxs) + 1

def get_mask_alpha(model0, method, mask_level, init_val):
    key_list = [key for key, _ in model0.named_modules()]
    max_layers_num = get_max_layers_num(key_list)
    mask_alphas = {}
    mask_bkp_alphas = {}
    
    for key in key_list:
        parent0, target0, target0_name = _get_submodules(model0, key)
        key_terms = key.split('.')
        if isinstance(target0, nn.Linear):
            if 'lm_head' in key:
                if mask_level == 'lblock3':
                    mask_bkp_alphas['lm_head'] = create_alpha(method, init_val)    
                    mask_alphas[key] = mask_bkp_alphas['lm_head']
                continue
            if mask_level == 'layerwise':
                if len(key_terms) > 2 and key_terms[1] == 'layers':
                    layer_id = key_terms[2]
                    if layer_id not in mask_bkp_alphas:
                        mask_bkp_alphas[layer_id] = create_alpha(method, init_val)    
                    mask_alphas[key] = mask_bkp_alphas[layer_id]
                else:
                    mask_alphas[key] = create_alpha(method, init_val)
            elif 'block3' in mask_level:
                if max_layers_num > 30:
                    layer_id_split1, layer_id_split2 = 11, 21
                else:
                    layer_id_split1, layer_id_split2 = 8, 16

                if len(key_terms) > 2 and key_terms[1] == 'layers':
                    layer_id = int(key_terms[2])
                    if layer_id < layer_id_split1:
                        block_id = 0
                    elif layer_id < layer_id_split2:
                        block_id = 1
                    else:
                        block_id = 2
                    if block_id not in mask_bkp_alphas:
                        mask_bkp_alphas[block_id] = create_alpha(method, init_val)
                    
                    mask_alphas[key] = mask_bkp_alphas[block_id]
                else:
                    mask_alphas[key] = create_alpha(method, init_val)
            elif 'block' in mask_level:
                block_n = int(mask_level[5:])
                bloch_each = int(max_layers_num / block_n)
                if len(key_terms) > 2 and key_terms[1] == 'layers':
                    layer_id = int(key_terms[2])
                    block_id = 0
                    for b_i in range(block_n):
                        block_id = b_i
                        if layer_id < bloch_each * (b_i + 1):
                            break
                    if block_id not in mask_bkp_alphas:
                        mask_bkp_alphas[block_id] = create_alpha(method, init_val)
                    mask_alphas[key] = mask_bkp_alphas[block_id]
                else:
                    mask_alphas[key] = create_alpha(method, init_val)
            elif mask_level == 'linearwise':
                mask_alphas[key] = create_alpha(method, init_val)
    return mask_alphas, mask_bkp_alphas
             
def warp_models_to_masked_interpolation(device, method, mask_level,
                                        model0, model1, init_val=0.2, epsilon=0):
    """
    Params:
        device: the device of the original params
        method: two methods in [mask_linear, mask_graft]
        mask_level: determine the mask type used in the interpolation, 
            e.g., mask_level = "layerwise", nn.linear in each layer share a same interpolation weight.
                  mask_level = "block3", split the model into 3 block and each block use one interpolation weight.
                  mask_level = "linearwise", each nn.linear contains its own interpolation weight.
        model0: the model used to wise
        model1: the model used to wise
        init_val: the init value for the interpolation weights
                (When it for graft it stand for the init propotion)
    """
    # model0 and model1 
    key_list = [key for key, _ in model0.named_modules()]
    mask_alphas, mask_bkp_alphas = get_mask_alpha(model0, method, mask_level, init_val)
    base_gammas = {}
    dtype = torch.bfloat16
    
    for key in key_list:
        parent0, target0, target0_name = _get_submodules(model0, key)
        parent1, target1, target1_name = _get_submodules(model1, key)
        key_terms = key.split('.')
        if isinstance(target0, nn.Linear): 
            if 'lm_head' in key and key not in mask_alphas:
                mask_alpha = create_alpha(method, init_val)
            else:
                mask_alpha = mask_alphas[key]

            if method == 'mask_norm_sigmoid_linear': 
                new_target0 = MaskNormalizedSigmoidLinear(target0.in_features, target0.out_features, target0.bias is not None)
                new_target0.set_wise_model(device, dtype, target0, target1, 
                                        wise_weight_alpha=mask_alpha, wise_bias_alpha=mask_alpha,
                                        normalized_alphas=list(mask_bkp_alphas.values()),
                                        init_val=init_val, epsilon=epsilon)
            setattr(parent0, target0_name, new_target0)
    return mask_alphas, mask_bkp_alphas, base_gammas

approaches = ['mask_linear', 'mask_graft']

class MaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        model0 = kwargs['model0']
        model1 = kwargs['model1']
        training_args = kwargs['args']
        self.approach = training_args.approach
        self.reg_alpha = training_args.reg_alpha
        self.sum_reg_type = training_args.sum_reg_type
        self.init_val = training_args.init_val
        self.mask_level = training_args.mask_level  
        self.epsilon = training_args.epsilon      
        self.device = training_args.device
        self.mask_alphas, self.mask_bkp_alphas, self.base_gammas = warp_models_to_masked_interpolation(self.device, self.approach, self.mask_level, model0, model1, self.init_val, self.epsilon)
        
        del model1
        kwargs['model'] = model0
        kwargs.pop('model0'), kwargs.pop('model1')
        
        super(MaskTrainer, self).__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if False:
            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls([self.mask_alphas[key] for key in self.mask_alphas], **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
    
        def normalize_alpha(alpha, alphas):
            epsilon = self.epsilon
            normalized_alpha = self.init_val * (F.sigmoid(alpha) + epsilon) / sum([F.sigmoid(alpha_) + epsilon for alpha_ in alphas]) * len(alphas)
            return normalized_alpha
        reg_loss = 0
        n_p = 0
        def reg_type_func(layer_id, reg_type):
            reg_type = str(reg_type)
            if reg_type.endswith('.2'):
                return 25 / (abs(layer_id - 13) + 1.0)
            else:
                return 1
        def reg_func(normalized_alpha, init_val, reg_type):
            reg_type = str(reg_type)
            if reg_type.startswith('0.'):
                # 0.0-0.9
                return torch.abs(normalized_alpha - init_val)
            elif reg_type.startswith('1.'):
                # 1.0-1.9
                return torch.sum((normalized_alpha - init_val)**2)
            elif reg_type.startswith('2.'):
                # 2.0-2.9
                return 1 + (normalized_alpha - init_val)
        for n, p in self.mask_alphas.items():
            if 'lm_head' in n:
                layer_id = 26
            else:
                layer_id = float(n.split('.')[2])
            if p.requires_grad:
                n_p += 1
                normalized_alpha = normalize_alpha(p, list(self.mask_bkp_alphas.values()))
                reg_loss += reg_type_func(layer_id, self.sum_reg_type) * (reg_func(normalized_alpha, self.init_val, self.sum_reg_type)).mean()
        reg_loss = reg_loss 
        
        loss = loss + self.reg_alpha * reg_loss
        
        outputs["loss"] = loss

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

        # Save Mask alphas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        torch.save({"mask_alphas":self.mask_alphas,
                    "mask_bkp_alphas":self.mask_bkp_alphas,
                    "init_val":self.init_val,
                    "epsilon":self.epsilon,
                    "base_gammas":self.base_gammas}, os.path.join(output_dir, 'mask_alphas.bin'))