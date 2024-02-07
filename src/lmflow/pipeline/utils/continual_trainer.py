
import copy

import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    Trainer
)

from transformers.trainer import *

if is_apex_available():
    from apex import amp

import deepspeed


def fisher_matrix_diag_bert_dil(t, train, device, model, criterion, sbatch=20):
    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()

    for i in tqdm(range(0, len(train), sbatch), desc='Fisher diagonal', ncols=100, ascii=True):
        b = torch.LongTensor(np.arange(i, np.min([i + sbatch, len(train)]))).cuda()
        batch = train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        input_ids, segment_ids, input_mask, targets, _ = batch

        # Forward and backward
        model.zero_grad()
        output_dict = model.forward(input_ids, segment_ids, input_mask)
        output = output_dict['y']

        loss = criterion(t, output, targets)
        loss.backward()
        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += sbatch * p.grad.data.pow(2)
    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / len(train)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
    return fisher


approaches = ["kd", "l1", "l2", "swa", "default"]

class ContinualTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(ContinualTrainer, self).__init__(*args, **kwargs)
        training_args = kwargs['args']
        self.approach = training_args.approach
        self.alpha = training_args.alpha
        if self.approach in ["kd", "l1", "l2"]:
            teacher_model = copy.deepcopy(self.model)
            self.ds_engine_teacher = deepspeed.initialize(model=teacher_model, config_params="examples/ds_config.json")[0]
            self.ds_engine_teacher.module.eval()
            for p in self.ds_engine_teacher.module.parameters():
                p.requires_grad = False
            self.teacher_params = {n: p for n, p in self.ds_engine_teacher.module.named_parameters()}
            # print(self.teacher_params)
            self.teacher_model = self.ds_engine_teacher.module
            self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
            self.temperature = 1

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_func = getattr(self, f"compute_loss_{self.approach}")
        results = loss_func(model, inputs, return_outputs=return_outputs)
        return results

    def compute_loss_kd(self, model, inputs, return_outputs=False):
        loss, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        with torch.no_grad():
            teacher_outputs = self.ds_engine_teacher(**inputs)
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs.hidden_states
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs.hidden_states
        kd_loss = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(s_logits / self.temperature, dim=-1),
                    nn.functional.softmax(t_logits / self.temperature, dim=-1),
                )
                * (self.temperature) ** 2
        )
        loss = loss + self.alpha * kd_loss
        student_outputs["loss"] = loss
        return (loss, student_outputs) if return_outputs else loss

    def compute_loss_l2(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        task_reg_loss = 0
        param_num = 0
        for n, p in model.module.named_parameters():
            if p.requires_grad:
                task_reg_loss += ((p - self.teacher_params[n]) ** 2).sum()
                param_num += torch.numel(p)
        loss = loss + self.alpha * task_reg_loss / param_num * 1e+9
        print(loss, 'task', task_reg_loss, 'actual', self.alpha * task_reg_loss / param_num * 1e+9)
        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss

    def compute_loss_l1(self, model, inputs, return_outputs=False):
        # print('running l1 loss')
        # print(model.state_dict())
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        task_reg_loss = 0
        param_num = 0 
        for n, p in model.module.named_parameters():
            if p.requires_grad:
                # if len(p.size()) > 0 and p.size()[0] > 0:
                #     print(p.size(), self.teacher_params[n])
                task_reg_loss += (torch.abs(p - self.teacher_params[n])).sum()
                param_num += torch.numel(p)
        # print('whole', task_reg_loss)
                
        loss = loss + self.alpha * task_reg_loss / param_num * 1e+5
        print(loss, 'task', task_reg_loss, 'actual', self.alpha * task_reg_loss / param_num * 1e+5)
        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss

    def compute_loss_swa(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def compute_loss_default(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
