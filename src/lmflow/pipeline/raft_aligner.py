#!/usr/bin/env python
# coding=utf-8
"""
The Aligner class simplifies the process of running alignment.
"""

import logging
import numpy as np
import os
import sys
import time
from itertools import chain
import json
import torch
import torch.distributed as dist
import transformers
from datasets import (
    set_caching_enabled,
    Dataset,
    DatasetDict,
)
from transformers import (
    default_data_collator,
    pipeline,
    set_seed,
    AutoModelForCausalLM)
from transformers.testing_utils import CaptureLogger

from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset as LMFlowDataset
from lmflow.pipeline.base_aligner import BaseAligner
# from lmflow.pipeline.utils.raft_trainer import RaftTrainer

logger = logging.getLogger(__name__)


class RaftAligner(BaseAligner):
    """
    Initializes the `RaftAligner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    raft_aligner_args : RaftAlignerArguments object.
        Contains the arguments required to perform alignment.

    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.

    """
    def __init__(self, model_args, data_args, aligner_args, *args, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = aligner_args

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.INF = 888888888
        logger.setLevel(logging.INFO)

        output_reward_path = aligner_args.output_reward_path
        if output_reward_path is not None:
            os.makedirs(os.path.dirname(output_reward_path), exist_ok=True)
            # Deletes a maybe-exist file
            try:
                os.remove(output_reward_path)
            except OSError:
                pass
        
        self.raft_infer_samples_store_dir = aligner_args.raft_infer_set + "/my_infer_set.json"
        self.raft_filter_samples_store_dir = aligner_args.raft_filtered_set + "/my_filtered_set.json"
        self.raft_rewards_store_dir = aligner_args.raft_exp_dir + "/reward_record.txt"
        set_seed(aligner_args.seed)

    def _initialize_trainer(self, model, tokenizer, training_args):
        """
        This function takes the model and tokenizer as the input and initialize the trainer.
        """
        trainer = RaftTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_dict({"text": [ " " ] }),
            eval_dataset=Dataset.from_dict({}),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )
        return trainer


    def _load_input_dataset(self, dataset, tokenizer):
        """
        Load input dataset (i.e. prompt/question dataset) for training.

        Args:
            dataset: A Dataset object.
                The dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        ds = dataset.get_backend_dataset()#.select(np.arange(8000))

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["text"])
            sample['input'] = sample["text"] #tokenizer.decode(sample["input_ids"])
            return sample
        if self.mode == "raft_get_rewards":
            pass
        else:
            ds = ds.map(tokenize, batched=False)
            ds = ds.filter(lambda x: len(x["input_ids"]) <= 256)
        
        ds.set_format(type='torch')

        return ds

    def _clean_text(self, text):
        if len(text) == 0:
            return text
        stext = [x for x in text.split("###Human") if x]
        #return stext[0].strip().strip("#") 
        return stext[0].strip().strip("#").replace("<s>", "").replace("</s>", "")

    def _discard_sample(self, text):
        if "#" in text:
            return True
        elif len(text) < 4: # delete empty sample
            return True
        return False
    def _get_batch_dataset_local(
            self,
            model,
            batch_input,
            K=8,
            local_rank=0,
            output_min_length=16,
            output_max_length=48,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            output_reward_path=None,
        ):
            """
            :param batch_input: input prompts
            """
            # we will get the batch dataset via Dataset.from_dict
            start_time = time.time()


            # We repeat each prompt for K times
            all_prompts = batch_input['input']
            querys = []
            for prompt in all_prompts:
                querys.extend([prompt for _ in range(K)])
            data_size = len(querys)
            assert data_size == K * len(all_prompts)

            input_texts = []

            all_texts_record = []
            all_responses_record = []

            for i, query in enumerate(querys):
                input_texts.append(query)

                if (i + 1) % infer_batch_size == 0 or (i+1 == data_size):
                    print(i, time.time()-start_time)

                    gen_len = np.random.randint(output_min_length, output_max_length)
                    generation_kwargs["max_new_tokens"] = gen_len
                    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_kwargs)
                    generated_texts = []
                    output_ids = outputs
                    input_ids = inputs['input_ids']
                    for j, output in enumerate(output_ids):
                        prompt_length = len(input_ids[j])
                        tgenerated_text = output[prompt_length:]
                        tgenerated_text = tokenizer.decode(tgenerated_text, skip_special_tokens=True)
                        generated_texts.append(tgenerated_text)

                    generated_texts = [
                        self._clean_text(generated_text) for generated_text in generated_texts
                    ]
                    #lengths = [len(txt) for txt in generated_texts]
                    #print(input_texts[0], "!!!!", generated_texts[0])

                    #if np.sum(lengths) == 0:
                    #    print("The model says nothing at all.")
                    #    if training_args.local_rank ==0:
                    #        print(input_texts[0], "!!!!", generated_texts[0])

                    input_texts = [txt.replace("<s>", "").replace("</s>", "") for txt in input_texts]
                    all_texts_record.extend(input_texts)
                    all_responses_record.extend(generated_texts)
                    input_texts = []
                    
                    #if i < 100:
                    #    print(all_texts_record[-1], generated_texts[-1])
            assert len(all_responses_record) == data_size

            data = []
            for i in range(len(all_prompts)):
                data.append({"input": all_texts_record[i * K], "output": all_responses_record[i * K : (i+1) * K]})


            world_size = int(os.getenv("WORLD_SIZE", "1"))
            all_process_data =[{}] * world_size
            dist.all_gather_object(all_process_data, data)
            
            # Communicate to gather the samples
            gathered_data = []
            for i in range(world_size):
                gathered_data.extend(all_process_data[i])

            output_eval_dataset = {}
            output_eval_dataset['type'] = 'text_only'
            output_eval_dataset['instances'] = gathered_data
            end_time = time.time()

            if training_args.local_rank == 0:
                logger.info(f"collected data of {len(gathered_data)}")
                print("it cost ", end_time-start_time)

                with open(self.raft_infer_samples_store_dir, 'w', encoding='utf8') as f:
                    json.dump(output_eval_dataset, f, ensure_ascii=False)

    '''
    def _get_batch_dataset_local(
            self,
            model,
            batch_input,
            K=8,
            local_rank=0,
            output_min_length=16,
            output_max_length=48,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            output_reward_path=None,
        ):
            """
            :param batch_input: input prompts
            """
            # we will get the batch dataset via Dataset.from_dict
            start_time = time.time()


            # We repeat each prompt for K times
            all_prompts = batch_input['input']
            querys = []
            for prompt in all_prompts:
                querys.extend([prompt for _ in range(K)])
            data_size = len(querys)
            assert data_size == K * len(all_prompts)

            input_texts = []

            all_texts_record = []
            all_responses_record = []

            for i, query in enumerate(querys):
                input_texts.append(query)

                if (i + 1) % infer_batch_size == 0 or (i+1 == data_size):
                    print(i, time.time()-start_time)

                    gen_len = np.random.randint(output_min_length, output_max_length)
                    generation_kwargs["max_new_tokens"] = gen_len
                    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_kwargs)
                    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    Tgenerated_texts = [
                        generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                    ]
                    lengths = [len(txt) for txt in Tgenerated_texts]
                    if np.sum(lengths) == 0:
                        print("The model says nothing at all TTTT.")
                    generated_texts = [
                        self._clean_text(generated_text) for generated_text in Tgenerated_texts
                    ]
                    lengths = [len(txt) for txt in generated_texts]
                    if np.sum(lengths) == 0:
                        print("The model says nothing at all.")
                        if training_args.local_rank ==0:
                            print(input_texts[0], "!!", Tgenerated_texts[0])

                    input_texts = [txt.replace("<s>", "").replace("</s>", "") for txt in input_texts]
                    all_texts_record.extend(input_texts)
                    all_responses_record.extend(generated_texts)
                    input_texts = []
                    
                    #if i < 100:
                    #    print(all_texts_record[-1], generated_texts[-1])
            assert len(all_responses_record) == data_size

            data = []
            for i in range(len(all_prompts)):
                data.append({"input": all_texts_record[i * K], "output": all_responses_record[i * K : (i+1) * K]})


            world_size = int(os.getenv("WORLD_SIZE", "1"))
            all_process_data =[{}] * world_size
            dist.all_gather_object(all_process_data, data)
            
            # Communicate to gather the samples
            gathered_data = []
            for i in range(world_size):
                gathered_data.extend(all_process_data[i])

            output_eval_dataset = {}
            output_eval_dataset['type'] = 'text_only'
            output_eval_dataset['instances'] = gathered_data
            end_time = time.time()

            if training_args.local_rank == 0:
                logger.info(f"collected data of {len(gathered_data)}")
                print("it cost ", end_time-start_time)

                with open(self.raft_infer_samples_store_dir, 'w', encoding='utf8') as f:
                    json.dump(output_eval_dataset, f, ensure_ascii=False)
    '''
    def add_noise(self, values):
        
        if np.random.random() < 0.2:

            my_means = [-0.75, -0.25, 0.5, 1]
            my_mean = np.random.choice(my_means)

            return values + np.random.normal(my_mean, 1.0)
        else:
            return values
    
    '''
    def _get_kl(self, query, responses, model, ref_model, tokenizer, device):
        """
        This function receives:
        query = ###Human:how are you? ###Assistant:
        responses = [Fine, Good, ..., Great] (K responses)
        we compute the conditional KL for each sample and return [KL-1, ..., KL-K]
        """
        kl_seq = []
        prob_seq = []
        #query = test_texts[i]
        with torch.no_grad():
            query_inputs = tokenizer(query, return_tensors="pt", padding=True).to(device)
            query_len = query_inputs['input_ids'].shape[1]
            #print(query_inputs, query_len)
            for i in range(len(responses)):
                one_response = responses[i]
                #res_input = tokenizer(one_response, return_tensors="pt", padding=True).to(0)
                inputs = tokenizer(query + one_response, return_tensors="pt", padding=True).to(device)

                logits = model(**inputs)['logits']
                ref_logits = ref_model(**inputs)['logits']

                input_ids = inputs["input_ids"]
            
                logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=2)
                logprobs = torch.gather(logp, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(-1)

                ref_logp = torch.nn.functional.log_softmax(ref_logits[:, :-1, :], dim=2)
                ref_logprobs = torch.gather(ref_logp, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(-1)

                
                # We compute the conditional KL only (p(y|x)) so we start from query_len - 1
                kl_pt = torch.sum(logprobs[:, query_len-1:] - ref_logprobs[:, query_len-1:], axis=1)
                #print(torch.sum(logprobs[:, query_len-1:] , axis=1))
                #prob_seq.append(torch.sum(ref_logprobs[:, query_len-1:], axis=1).item())
                kl_seq.append(kl_pt.item())
        
        print(kl_seq, "kl")
        #print(prob_seq, "log_prob")
        return kl_seq#, prob_seq

    def _raft_get_rewards(
        self,
        batch_input,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
        model=None,
        ref_model=None,
        tokenizer=None,
    ):
        """
        GET KL
        """
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output'] 
        K = len(responses[0])
        #K = 1
        data = []
        all_rewards = []
        gold_reward_train = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]
            iter_kl = self._get_kl(querys[i], responses[i], model, ref_model, tokenizer, training_args.device)

  
            rewards = iter_kl
            gold_rewards = iter_kl
            record_reward  = np.mean(rewards) #rewards[0]
            reward_eva.append(record_reward)
            all_rewards.append(gold_rewards)

            # we impose some post-detection and discard the samples with certain criteria.

            idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] >= -1000:
                #data.append({"input": q, "output": tmp_responses, "rewards": rewards})
                data.append({"text": q + tmp_responses[idx_to_record]})

                reward_train.append(rewards[idx_to_record])                
                gold_reward_train.append(gold_rewards[idx_to_record])
        

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)
        all_rewards_mean = np.mean(all_rewards)
        mean_gold_train_reward = np.mean(gold_reward_train)
        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i], all_rewards_mean, mean_gold_train_reward] for i in range(len(data))]
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_gold_reward = []
        gathered_gold_train_reward = []

        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)
    
            tmp_gold_reward = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_gold_reward.extend(tmp_gold_reward)

            tmp_gold_train_reward = [tmp[4] for tmp in all_process_list[i]['data']]
            gathered_gold_train_reward.extend(tmp_gold_train_reward)

        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if training_args.local_rank == 0:
            if "no_store" in self.raft_filter_samples_store_dir:
                pass
            else:
                with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                    json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "   "  
                + str(np.mean(gathered_gold_reward)) + "   " + str(np.mean(gathered_gold_train_reward)) + "   " + "\n")

    '''
    
    def _raft_get_rewards(
        self,
        batch_input,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output'] 
        K = len(responses[0])
        #K = 1
        data = []
        all_rewards = []
        gold_reward_train = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]
            #texts_for_rewards = [q + tmp_responses[0]]
            #if i < 3:
            #    print(texts_for_rewards)
            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            gold_rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
            #rewards = [ self.add_noise(sample["value"]) for sample in reward_dataset.to_dict()["instances"] ] 
            #(rewards)
            rewards = gold_rewards
            
            record_reward  = np.mean(rewards) #rewards[0]
            reward_eva.append(record_reward)
            all_rewards.append(gold_rewards)

            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -self.INF

            idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] >= -1000:
                #data.append({"input": q, "output": tmp_responses, "rewards": rewards})
                data.append({"text": q + tmp_responses[idx_to_record]})

                reward_train.append(rewards[idx_to_record])                
                gold_reward_train.append(gold_rewards[idx_to_record])
        

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)
        all_rewards_mean = np.mean(all_rewards)
        mean_gold_train_reward = np.mean(gold_reward_train)
        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i], all_rewards_mean, mean_gold_train_reward] for i in range(len(data))]
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_gold_reward = []
        gathered_gold_train_reward = []

        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)
    
            tmp_gold_reward = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_gold_reward.extend(tmp_gold_reward)

            tmp_gold_train_reward = [tmp[4] for tmp in all_process_list[i]['data']]
            gathered_gold_train_reward.extend(tmp_gold_train_reward)

        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if training_args.local_rank == 0:
            if "no_store" in self.raft_filter_samples_store_dir:
                pass
            else:
                with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                    json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "   "  
                + str(np.mean(gathered_gold_reward)) + "   " + str(np.mean(gathered_gold_train_reward)) + "   " + "\n")

    
    '''
    def _raft_get_rewards(
        self,
        batch_input,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output'] 
        K = len(responses[0])
        #K = 1
        data = []
        all_rewards = []
        gold_reward_train = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]
            #texts_for_rewards = [q + tmp_responses[0]]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            gold_rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
            #rewards = [ self.add_noise(sample["value"]) for sample in reward_dataset.to_dict()["instances"] ] 
            #(rewards)
            rewards = gold_rewards
            
            record_reward  = np.mean(rewards) #rewards[0]
            reward_eva.append(record_reward)
            all_rewards.append(gold_rewards)

            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -5

            idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] >= -1000:
                data.append({"input": q, "output": tmp_responses, "rewards": rewards})
                #data.append({"text": q + tmp_responses[idx_to_record]})

                reward_train.append(rewards[idx_to_record])                
                gold_reward_train.append(gold_rewards[idx_to_record])
        

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)
        all_rewards_mean = np.mean(all_rewards)
        mean_gold_train_reward = np.mean(gold_reward_train)
        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i], all_rewards_mean, mean_gold_train_reward] for i in range(len(data))]
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_gold_reward = []
        gathered_gold_train_reward = []

        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)
    
            tmp_gold_reward = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_gold_reward.extend(tmp_gold_reward)

            tmp_gold_train_reward = [tmp[4] for tmp in all_process_list[i]['data']]
            gathered_gold_train_reward.extend(tmp_gold_train_reward)

        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if training_args.local_rank == 0:
            if "no_store" in self.raft_filter_samples_store_dir:
                pass
            else:
                with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                    json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "   "  
                + str(np.mean(gathered_gold_reward)) + "   " + str(np.mean(gathered_gold_train_reward)) + "   " + "\n")

    '''
    
    
    '''
    def _raft_get_rewards(
        self,
        batch_input,
        training_args=None,
        reward_model=None,
        output_reward_path=None,
    ):
        """
        This function computes the rewards for the K responses given each prompt.
        We also collect the best of K samples into a filtered dataset.
        """
        start_time = time.time()
        reward_eva = []
        reward_train = []
        querys = batch_input['input']
        responses = batch_input['output'] 
        K = len(responses[0])
        #K = 1
        data = []
        all_rewards = []
        gold_reward_train = []
        for i in range(len(querys)):
            q = querys[i]
            tmp_responses = responses[i]
            texts_for_rewards = [q + r for r in tmp_responses]
            #texts_for_rewards = [q + tmp_responses[0]]

            texts_for_reward_dataset = LMFlowDataset.create_from_dict({
                "type": "text_only",
                "instances": [
                    { "text": text } for text in texts_for_rewards
                ],
            })

            reward_dataset = reward_model.inference(texts_for_reward_dataset)
            gold_rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]
            #rewards = [ self.add_noise(sample["value"]) for sample in reward_dataset.to_dict()["instances"] ] 
            #(rewards)
            rewards = gold_rewards
            
            record_reward  = rewards[0]
            reward_eva.append(record_reward)
            all_rewards.append(gold_rewards)

            # we impose some post-detection and discard the samples with certain criteria.
            for kk in range(K):
                if self._discard_sample(tmp_responses[kk]):
                    rewards[kk] = -self.INF

            idx_to_record = np.argmax(rewards)
            
            
            # if we discard all the samples, we do not record the sample 
            if rewards[idx_to_record] >= -1000:
                data.append({"text": q + tmp_responses[idx_to_record]})
                reward_train.append(rewards[idx_to_record])                
                gold_reward_train.append(gold_rewards[idx_to_record])
        

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size

        mean_eval_reward = np.mean(reward_eva)
        all_rewards_mean = np.mean(all_rewards)
        mean_gold_train_reward = np.mean(gold_reward_train)
        data_to_send = {
            'data': [[data[i], mean_eval_reward, reward_train[i], all_rewards_mean, mean_gold_train_reward] for i in range(len(data))]
        }
        dist.all_gather_object(all_process_list, data_to_send)
        gathered_data = []
        gathered_reward = []
        gathered_train_reward = []
        gathered_gold_reward = []
        gathered_gold_train_reward = []

        for i in range(world_size):
            tmp_data = [tmp[0] for tmp in all_process_list[i]['data']]
            gathered_data.extend(tmp_data)

            tmp_reward = [tmp[1] for tmp in all_process_list[i]['data']]
            gathered_reward.extend(tmp_reward)

            tmp_train_reward = [tmp[2] for tmp in all_process_list[i]['data']]
            gathered_train_reward.extend(tmp_train_reward)
    
            tmp_gold_reward = [tmp[3] for tmp in all_process_list[i]['data']]
            gathered_gold_reward.extend(tmp_gold_reward)

            tmp_gold_train_reward = [tmp[4] for tmp in all_process_list[i]['data']]
            gathered_gold_train_reward.extend(tmp_gold_train_reward)

        logger.info(f"collected data of {len(gathered_data)}")
        logger.info(f"mean reward: {np.mean(gathered_reward)}, reward in train set: {np.mean(gathered_train_reward)}")
        

 
        # We store the training set for monitoring the RAFT training
        output_eval_dataset = {}
        output_eval_dataset['type'] = 'text_only'
        output_eval_dataset['instances'] = gathered_data
        import json
        if training_args.local_rank == 0:
            with open(self.raft_filter_samples_store_dir, 'w', encoding='utf8') as f:
                json.dump(output_eval_dataset, f, ensure_ascii=False)

            with open(self.raft_rewards_store_dir, 'a') as f:
                f.write(str(np.mean(gathered_reward)) + "   " + str(np.mean(gathered_train_reward)) + "   "  
                + str(np.mean(gathered_gold_reward)) + "   " + str(np.mean(gathered_gold_train_reward)) + "   " + "\n")

    '''
    def align(self, model, dataset, reward_model):
        """
        Perform alignment for a model

        Parameters
        ------------
        model : BaseModel object.
        dataset: Dataset object.
            Input dataset for model to generate outputs. The input and output
                will then be feed into reward model to get the reward for
                alignment.
        reward_model: RegressionModel object.
        """
        tokenizer = model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        # set_caching_enabled(False)

        wrapped_model = model
        model = model.get_backend_model()

        aligner_args = self.aligner_args
        training_args = aligner_args
        model_args = self.model_args
        data_args = self.data_args
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.mode = aligner_args.mode
        self.seed = aligner_args.raft_random_seed

        generation_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature": training_args.output_temperature,
        }

        set_seed(42 + training_args.local_rank)

        K = int(1/aligner_args.top_reward_percentage)
        M = int(aligner_args.raft_batch_size / world_size)
        self.store_dir = aligner_args.output_dir
        print(M, K)
        dataset = self._load_input_dataset(dataset, tokenizer)

        data_size = len(dataset['input'])
        share = int(len(dataset) / world_size) 
        dataset = dataset.select(np.arange(training_args.local_rank * share, (training_args.local_rank + 1)*share))


        raft_trainer = self._initialize_trainer(model, tokenizer, training_args)
        raft_trainer.train(resume_from_checkpoint=False, is_first_time=True)
        mode = aligner_args.mode

        ####
        #ref_model = AutoModelForCausalLM.from_pretrained("/home/xiongwei/rm_study/LMFlow/output_models/best4/proxy_sft_open_llama_3b_v2_2epoch_helpful_and_harmless")

        #raft_trainer2 = self._initialize_trainer(ref_model, tokenizer, training_args)
        #raft_trainer2.train(resume_from_checkpoint=False, is_first_time=True)
        

        #####
        if mode == "raft_get_samples":
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            data_size = len(dataset['input'])
            idxs = np.arange(data_size)
            end_idx = np.min([data_size, (training_args.iter_id + 1) * M])
            batch_input = shuffled_dataset.select(idxs[training_args.iter_id * M : end_idx])
            model.gradient_checkpointing_disable()
            model.config.use_cache = True

            start_time = time.time()
    
            selected_dataset = self._get_batch_dataset_local(
                raft_trainer.tmp_model,
                batch_input,
                K,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=aligner_args.inference_batch_size_per_device,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                output_reward_path=aligner_args.output_reward_path,
            )
            end_time = time.time()
            logger.info("It takes %.2f s to inference one stage", end_time - start_time)
        elif mode == "raft_get_rewards":
            batch_input = dataset
            start_time = time.time()

            selected_dataset = self._raft_get_rewards(
                batch_input=batch_input,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
                #model=raft_trainer.tmp_model,
                #ref_model=ref_model,
                #tokenizer=tokenizer,
            )
            end_time = time.time()
            logger.info("It takes %.2f s to inference one stage", end_time - start_time)
        else:
            print("The mode is not supported...")

        return None 
