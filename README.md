# Implementation of ['Mitigating the Alignment Tax of RLHF'](https://arxiv.org/abs/2309.06256)

This is the repository of implementation of Adaptive Model Averaging in ['Mitigating the Alignment Tax of RLHF'](https://arxiv.org/abs/2309.06256). In this paper we explore model averaging, which interpolates between pre and post RLHF model weights, to achieve a more efficient reward-tax Pareto front. Building on the analysis and the observation that averaging different layers of the transformer leads to significantly different reward-tax trade-offs, we propose Adaptive Model Averaging (AMA) to adaptively find various combination ratios of model layers. AMA seeks to maximize the alignment reward while incurring minimal alignment tax. Moreover, we validate AMA’s performance across a range of RLHF algorithms over OpenLLaMA-3B and further extend our findings to Mistral-7B.
We use the LMFlow as the framework which is compatible with other huggingface models. 

<!-- 
<p align="center" width="100%">
<img src="assets/raft.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p> -->

## Quick Start

Before training or inference, you need to install the package as follows.
### Setup

```bash
git clone 
cd 
conda create -n ama_env python=3.9 -y
conda activate ama_env
conda install mpi4py
./install.sh
```


### Prepare Dataset
Since many experiments of our papers are conducted on HH-RLHF dataset. We have prepared the [HH-RLHF](https://github.com/anthropics/hh-rlhf) dataset and preprocess it into SFT, RM, and RLHF datasets. You can download them by running 

```bash
cd data
./download.sh hh_rlhf
```

Each sample of the dataset consists of a prompt (chat history) and two responses, where on is preferred compared to the other one. An example of the dataset is

**###Human**: How can I get my girlfriend to cook more?
**###Assistant**: Have you tried reminding her of how nice the food tastes?
**###Human**: I could do it more.
**###Assistant**:

**Responses**:
- Preferred $y_w$: That might work.  But you could also explain to her how much you enjoy her cooking and how much you appreciate her effort.  Try pointing out the activities that you think go into making great meals: sourcing the ingredients, cleaning and preparing the kitchen, etc.
- Rejected $y_l$: Have you considered making an effort to create more harmonious interactions?

We use the preferred responses of HH-RLHF training set as the SFT dataset, use the complete ccomparison data (training + test) for reward modeling, and use the prompts of the training set for RLHF.

### Model Averaging
We first provide the script for make general model averaging between two models.

```bash
bash scripts/postprocess/weight_interpolation.sh
```
To make it work, you need to change the parameters below in the script 'scripts/postprocess/weight_interpolation.sh':

```bash
alpha=0.5
model_path0=path/to/model/before/rlhf
model_path1=path/to/model/after/rlhf
...
weight_ensamble_save_path=path/to/save/ma_${alpha}_tag0_tag1
```
here the tag0 and tag1 can be used to specify the model0 and model1.
If the model0 with $\theta_0$, model1 with $\theta_1$, the save model will have the model weights of $\alpha * \theta_0 + (1 - \alpha) * \theta_1$, i.e., $\alpha=0$ means the model1 and $\alpha=1$ means the model0.

### Partwise Model Averaging
To leverage the parwise model averaging to repreduce the results in Section 4. we can still use the script 'scripts/postprocess/weight_interpolation.sh' but change the name of the weight_ensamble_save_path like this:
```bash

weight_ensamble_save_path=path/to/save/pma_${alpha}_${tag0}_split|10#20|0.4|0.3|0.2_${tag1}
```
alpha here only means the alpha weight of the lm_head layer but not other layers in transformer. tag0 and tag1 still represent the model0 and model1. 'split' means the merge method so just keep it here. '10#20' means we split the whole transformer layers into three part 0-10 (contain layer 10) is the first block, 11-20 (contain layer 20) is the second, and 21-final layer is the third. '0.4|0.3|0.2' represent the alpha weights of these three blocks. Actually you can extend the three block setting to arbitray blocks and just make (the number of alpha weights) = (the number of layer idx pivots) + 1.

Reminders: Since we parse the save name to get information, so make sure that there are no '|', '#', '_' inside your tag0 and tag1.

### Adaptive Model Averaging
To implement the adaptive model averaging, there are two steps: 1). optimization to get the alpha weights, 2). averaging based on the weights.

#### Optimization
```bash
bash scripts/mask_post/run_mask_finetune_raft.sh
```

Hyper-parameters of this optimziation process can be found in the script:

```bash
approach=mask_norm_sigmoid_linear  # the method used to average, just keep it
mask_level=block3  # split the transfromers into 3 blocks, it will automatically compute the layer idx pivots.
lr=2e-5  # learning rate of the optimzation process
init_val=0.2  # base alpha weight 
reg_alpha=1e-4  # the penalty of the regularization term 
sum_reg_type=0.2  # actually there are only 0, 0.2 two types 0 means direct l1 penalty, 0.2 means a weighted l1 penalty
epsilon=0.2  # epsilon value add on the normalization part, it can be used to control the whole variation.
```
there are also several paths variables you need to adjust to your own paths.

```bash
model_path=${project_dir}/path/to/after/rlhf
dataset_path=${project_dir}/path/to/data/collected
```

#### Averaging
```bash
bash scripts/postprocess/weight_mask_merge.sh
```

```bash
model_path0=${project_dir}/path/to/before/rlhf
model_path1=${project_dir}/path/to/after/rlhf

alphas_path=${project_dir}/path/to/mask_alphas.bin
weight_ensamble_save_path=${project_dir}/path/to/save
```
Averaging is almost the same as the script of model averaging, but here you just need to adjust paths of models with the learned mask_alphas.bin.

### Evaluations
We give the usage of the evaluations scripts of our experiments. All scripts will require model paths, we do not specify here.

#### Common Sense
We invoke the [lm-evluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evluate. (So you need to download the repo first.) 

```bash
bash scripts/eval_cs_qa/run_evaluation.sh
```

#### Drop/Squad/WMT14
We invoke [opencompass](https://github.com/open-compass/opencompass) and [lmflow_bencmark](https://github.com/shizhediao/forgetting-bench).
```bash
bash scripts/eval_drop_squad_wmt/run-evaluation-drop.sh
bash scripts/eval_drop_squad_wmt/run-evaluation-squad.sh
bash scripts/eval_drop_squad_wmt/run-evaluation-wmt.sh
```

#### PairRM Value

```bash
bash scripts/eval_pairrm/run_pairrm.sh
```

#### Reward Value

```bash
bash scripts/eval_raft/run_eval_raft_align.sh
```


## Support

If you need any help, please submit a Github issue.
## License
The code included in this project is licensed under the [Apache 2.0 license](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE).
If you wish to use the codes and models included in this project for commercial purposes, please sign this [document](https://docs.google.com/forms/d/e/1FAIpQLSfJYcci6cbgpIvx_Fh1xDL6pNkzsjGDH1QIcm4cYk88K2tqkw/viewform?usp=pp_url) to obtain authorization.

## Citation
If you find this repository useful, please consider giving ⭐ and citing our [paper](https://arxiv.org/abs/2309.06256):

```
@article{lin2024mitigating,
      title={Mitigating the Alignment Tax of RLHF}, 
      author={Lin, Yong and Lin, Hangyu and Xiong, Wei and Diao,  Shizhe and Liu, Jianmeng and Zhang, Jipeng and Pan, Rui and Wang, Haoxiang and Hu, Wenbin and Zhang, Hanning and Dong, Hanze and Pi, Renjie and Zhao, Han and Jiang, Nan and Ji, Heng and Yao, Yuan and Zhang, Tong},
      journal={arXiv preprint arXiv:2309.06256},
      year={2023}
}
```