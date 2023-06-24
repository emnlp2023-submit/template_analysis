Source code for the paper "Don't Underestimate the Template: Input Format Matters for In-Context Learning"

# Installation

Create a new conda environment:

```
conda create --name templates python=3.8
conda activate templates
```

Install the dependencies:

```
pip install -r requirements.txt
```

# Evaluation

In order to run experiments with LLaMA, you need to accept [its license](https://github.com/facebookresearch/llama/blob/main/LICENSE) and download the weights after approval. To convert the weights into the Transformers format, use [this guide](https://huggingface.co/docs/transformers/main/model_doc/llama).

`cache_dir` is an optional argument for a path to your HuggingFace cache. You may not specify it but you'll have to if you have LLaMA weights. You may also need to rename your weights directory or modify `utils.py` as it supposes that LLaMA weights are located at `{cache_dir}/llama-[7/13/30/65]b`.

Note that you may need to change `eval_batch_size` depending on a particular model and your hardware.

You may need to use more than 1 GPU for some models. In this case, you may also want to set `--device_map balanced_low_0` (see [HuggingFace documentation](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) for details)

## Baseline results

Standard random-2/4/8-shot learning with the Direct prediction method:

```
python prediction_methods.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --seed 59 13 21 \
  --num_shots [2/4/8] \
  --save_dir {save_path} \
  --cache_dir {hf_cache_path} \
  --wandb_entity {your_wandb_account} \
  --prediction_method direct_False \
  --eval_batch_size 16
```

## Prediction methods

```
python prediction_methods.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --seed 59 13 21 \
  --num_shots 2 \
  --save_dir {save_path} \
  --cache_dir {hf_cache_path} \
  --wandb_entity {your_wandb_account} \
  --prediction_method [direct_False/channel_True/calibrate_True] \
  --eval_batch_size 16
```

## Example selection methods 

```
python k_shot.py \
  -d [sst2/dbpedia/agnews/trec] \
  -m {model} \
  --examples_path selected_examples/{method}/{dataset} \
  --seed 59 13 21 \
  --num_shots [2/4/8] \
  --save_dir {scores_save_path} \
  --cache_dir {hf_cache_path} \
  --wandb_entity {your_wandb_account} \
  --prediction_method [direct_False/calibrate_True/channel_True] \
  --examples_selection_method [random/implicitly_topic_models/z-ICL] \
  --eval_batch_size 16
```

# Template transfer between setups

You need to get scores for all the setups, which are you interested in, with the following command:

```
templates.py \
  -m {model} \
  --dataset [sst2/dbpedia/agnews/trec] \
  --eval_batch_size 16 \
  --num_shots 0 \
  --save_dir {scores_save_path} \
  --cache_dir {hf_cache_path} \
  --prediction_method [direct_False/calibrate_True/channel_True]
```

# Template ensembles

You can get results for template ensembles with the following command:

```
templates_ensemble.py \
  -m {model} \
  -dataset [sst2/dbpedia/agnews/trec] \
  --num_templates 10 \
  --num_shots 2 \
  --prediction_method direct_False \
  --eval_batch_size 16 \
  --save_dir {save_path} \
  --cache_dir {hf_cache_path} \
  --seed 13 21 59
```

# Results analysis

After you run all the scripts you need, download your Weights & Biases runs to `all_runs.csv` and run `analyse_results.ipynb`
