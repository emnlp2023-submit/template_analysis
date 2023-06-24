import re
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from datasets import load_dataset

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
)


names_to_checkpoints = {'gpt2-large': 'gpt2-large',
                        'gpt2-xl': 'gpt2-xl',
                        'gptj': 'EleutherAI/gpt-j-6B',
                        'gpt-neox': 'EleutherAI/gpt-neox-20b',
                        'opt-1.3b': 'facebook/opt-1.3b',
                        'opt-6.7b': "facebook/opt-6.7b",
                        'opt-30b': "facebook/opt-30b",
                        'opt-66b': "facebook/opt-66b",
                        'bloom-1.7b': 'bigscience/bloom-1b7',
                        'bloom-3b': 'bigscience/bloom-3b',
                        'bloom-7.1b': 'bigscience/bloom-7b1',
                        'pythia-6.9b': 'EleutherAI/pythia-6.9b',
                        'pythia-12b': 'EleutherAI/pythia-12b',
                        'cerebras-6.7b': 'cerebras/Cerebras-GPT-6.7B',
                        'cerebras-13b': 'cerebras/Cerebras-GPT-13B',
                        'llama-7b': 'llama-7b',
                        'llama-13b': 'llama-13b',
                        'llama-30b': 'llama-30b',
                        'llama-65b': 'llama-65b',
                        'falcon-1b': 'tiiuae/falcon-rw-1b',
                        'falcon-7b': 'tiiuae/falcon-7b',
                        'falcon-40b': 'tiiuae/falcon-40b',
}



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help='dataset name.', nargs='+',
                        choices=['emotion', 'financial_phrasebank', 'poem_sentiment',
                                 'sst2', 'dbpedia', 'agnews', 'trec'])
    parser.add_argument('-m', '--model', '--inference_models', type=str, nargs='*',
                        help='Specify which models to run on instead of full names_to_checkpoints.')
    parser.add_argument("--num_shots", type=int, help='number of examples for ICL.', default=0)
    parser.add_argument("--num_templates", type=int, help='number of templates for templates ensemble', default=10)
    parser.add_argument("--seed", help='Seed for reproducibility.', type=int, default=59, nargs='+')
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="When not set creates a batch of size N_labels for each test input.")
    parser.add_argument("--examples_selection_method", default='random', nargs='*',
                        help="method for selecting examples for ICL.")
    parser.add_argument("--example_ids", type=int, default=None, nargs="+",
                        help="ids of the train samples to use as examples for ICL.")
    parser.add_argument("--examples_path", type=str, default=None,
                        help="specify path to json where the retrieved examples are stored")
    parser.add_argument("--prediction_method", default='direct_False', nargs='*',
                        help="Method of prediction on test inputs: generate takes a last token from a generation output"
                             ". direct compares model's losses over all possible labels. channel calculates losses "
                             "over inputs giving various possible labels. calibrate makes models predictions equal for "
                             "a context-free input.")
    parser.add_argument("--labels_loss", action='store_true',
                        help="Whether to calculate loss over whole sequence or only on the label part.")
    parser.add_argument("--precision", choices=['fp16', 'fp32'], default='fp16',
                        help='floating point precision for inference model.')
    parser.add_argument("--cache_dir", default="/mnt/data/hf_models", help="Path to huggingface cache")
    parser.add_argument("--save_dir", type=str, default=".", help="Where to save results")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_project", default='ExamplesSelection')
    parser.add_argument("--local_files_only", action='store_true', help="turn this on if you have already pre-downloaded weights for your model and don't want to endullating the storage or fruitlessly spending minutes of your precious time.")
    parser.add_argument("--device_map", default="auto")
    
    args = parser.parse_args()
    args.inference_models = args.model
    return args


class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def __repr__(self):
        return self.model.__repr__()
    def __str__(self):
        return self.model.__str__()

class TensorDataset(Dataset):
    def __init__(self, test_samples, tokenizer, labels, template, examples=None, examples_for_each_input=False,
                 method='direct', max_length=1024):
        if examples is None:
            examples = []

        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        self.inp_verbalizer, self.out_verbalizer, self.sep, self.big_sep = template
        if "{}" not in self.inp_verbalizer:
            raise ValueError("inp_verbalizer must contain {} for formatting")
        if "{}" not in self.out_verbalizer:
            raise ValueError("out_verbalizer must contain {} for formatting")
        self.tokenizer = tokenizer
        self.labels = labels
        self.examples = examples
        self.method = method
        self.max_length = max_length

        if examples_for_each_input:
            self.context = [self.add_examples_to_context(one_example, method) for one_example in examples]
        else:
            self.context = [self.add_examples_to_context(examples, method) for _ in range(len(test_samples))]
        if self.context[0]:
            self.context = [x + self.big_sep for x in self.context]
            print(self.context[:2])

        if self.tokenizer.bos_token_id is not None:
            self.context_tokenized = [[self.tokenizer.bos_token_id] for _ in range(len(self.context))]
        else:
            self.context_tokenized = [[] for _ in range(len(self.context))]
        for i in range(len(self.context)):
            self.context_tokenized[i].extend(tokenizer(self.context[i], add_special_tokens=False)['input_ids'])

        for i, input_text in tqdm(enumerate(test_samples)):
            for label in labels:
                input_ids, attention_mask, token_type_ids = self.prepro_one_sentence(input_text, label, i)

                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)
                self.token_type_ids.append(token_type_ids)

    def add_examples_to_context(self, examples, method):
        if 'channel' in method:
            return self.big_sep.join([f"{self.out_verbalizer.format(x[1])}{self.sep}{self.inp_verbalizer.format(x[0])}"
                                      for x in examples])
        else:
            return self.big_sep.join([f"{self.inp_verbalizer.format(x[0])}{self.sep}{self.out_verbalizer.format(x[1])}"
                                      for x in examples])

    def prepro_one_sentence(self, input_text, label, i):
        input_text = self.inp_verbalizer.format(input_text)
        label = self.out_verbalizer.format(label)
        if self.method == 'channel':
            label, input_text = input_text, label

        input_tokenized = self.tokenizer(input_text, add_special_tokens=False)['input_ids']
        sep_tokenized = self.tokenizer(self.sep, add_special_tokens=False)['input_ids']
        out_tokenized = self.tokenizer(label, add_special_tokens=False)['input_ids']
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        input_ids = self.context_tokenized[i] + input_tokenized + sep_tokenized + out_tokenized + eos
        begin = len(self.context_tokenized[i]) + len(input_tokenized) + len(sep_tokenized)
        end = len(input_ids) - 1
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * begin + [1] * (end - begin) + [0]

        to_predict = self.tokenizer.decode(input_ids[begin:end]).strip()
        gt = self.tokenizer.decode(out_tokenized).strip()
        assert to_predict == gt

        return input_ids, attention_mask, token_type_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.input_ids[idx]),
                "attention_mask": torch.tensor(self.attention_mask[idx]),
                "token_type_ids": torch.tensor(self.token_type_ids[idx])}



def load_split_dataset(dataset_name, seed=59, cache_dir='~/.cache/huggingface/datasets'):
    target_col = 'label'
    val_split = 'validation'
    if dataset_name == 'sst2':
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        labels_mp = dict(enumerate(['negative', 'positive']))
        input_col = 'sentence'
    elif dataset_name == 'dbpedia':
        dataset = load_dataset('dbpedia_14', cache_dir=cache_dir)
        labels_mp = dict(enumerate(["Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
                                    "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
                                    "Plant", "Album", "Film", "Written Work"]))
        input_col = 'content'
        val_split = 'test'
    elif dataset_name == 'agnews':
        dataset = load_dataset('ag_news', cache_dir=cache_dir)
        labels_mp = dict(enumerate(["World", "Sports", "Business", "Technology"]))
        input_col = 'text'
        val_split = 'test'
    elif dataset_name == 'trec':
        dataset = load_dataset('trec', cache_dir=cache_dir)
        labels_mp = dict(enumerate(["Description", "Entity", "Expression", "Human", "Location", "Number"]))
        input_col = 'text'
        target_col = 'coarse_label'
        val_split = 'test'
    else:
        raise NotImplementedError(f"Not implemented for {dataset_name}")

    train = pd.DataFrame({
        'input': dataset['train'][input_col],
        'target': dataset['train'][target_col]
    })
    train['target'] = train.target.map(labels_mp)
    if dataset_name == 'financial_phrasebank':
        train, val = train_test_split(train, test_size=.2, random_state=seed)
    elif dataset_name in ['dbpedia', 'agnews']:
        _, val = train_test_split(dataset[val_split], test_size=1000, random_state=seed)
        val = pd.DataFrame({
            'input': val[input_col],
            'target': val[target_col]
        })
        val['input'] = val.input.apply(lambda x: re.sub('[{,}]', '', x.strip()))
    else:
        val = pd.DataFrame({
            'input': dataset[val_split][input_col],
            'target': dataset[val_split][target_col]
        })
    val['target'] = val.target.map(labels_mp)

    return train, val, labels_mp


def generate(generator, prompts):
    res = generator(prompts, max_new_tokens=2, do_sample=False, batch_size=len(prompts))
    return [r[0]['generated_text'].split()[-1] for r in res]


@torch.inference_mode()
def get_loss(generator, batch, labels_loss=False):
    model = generator.model
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)

    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    token_type_ids = batch['token_type_ids']
    labels = torch.where(attention_mask == 1, input_ids, -100)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[..., :-1, :].contiguous().to(model.dtype)
    shift_labels = labels[..., 1:].contiguous().to(logits.device)
    losses = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(logits.size(0), logits.size(1))
    if labels_loss:
        label_mask = token_type_ids[..., 1:].contiguous().to(model.device)
        losses = losses * label_mask
        losses = losses.sum(dim=-1) / label_mask.sum(dim=-1)
    else:
        losses = losses.mean(dim=-1)
    losses = losses.detach().cpu()
    torch.cuda.empty_cache()
    return losses


def classify(losses, labels, p_cf=None, mode="diagonal_W", return_probs=False):
    num_classes = len(labels)
    if p_cf is None:
        # do not calibrate
        W = torch.eye(num_classes)
        b = torch.zeros(num_classes)
    else:
        # calibrate
        if mode == "diagonal_W":
            W = torch.linalg.inv(torch.eye(num_classes) * p_cf)
            b = torch.zeros(num_classes)
        elif mode == "identity_W":
            W = torch.eye(num_classes)
            b = -1 * p_cf[:, None]
        else:
            raise NotImplementedError(f"{mode} is not implemented for calibration")

    uncalibrated_probs = softmax(-losses)
    calibrated_probs = torch.matmul(uncalibrated_probs, W) + b

    if return_probs:
        return np.array(labels)[calibrated_probs.argmax(1)], calibrated_probs

    return np.array(labels)[calibrated_probs.argmax(1)]


def predict(generator, eval_dataset, labels, batch_size=1, method='direct', labels_loss=False,
            calibrate_dataset=None, mode='diagonal_W', return_probs=False, skip_cuda_oom=False):
    collator = DataCollatorForLanguageModeling(generator.tokenizer, mlm=False)

    if method == 'calibrate':
        torch.cuda.empty_cache()
        calibrate_dataloader = DataLoader(calibrate_dataset, shuffle=False, batch_size=batch_size, collate_fn=collator)
        cf_losses = []
        for batch in tqdm(calibrate_dataloader):
            cf_losses.extend(get_loss(generator, batch, labels_loss))
        cf_losses = torch.tensor(cf_losses, dtype=torch.float32).reshape(-1, len(labels))
        cf_label_probs = softmax(-cf_losses)
        p_cf = torch.mean(cf_label_probs, dim=0)
        torch.cuda.empty_cache()
    else:
        p_cf = None

    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=collator)
    losses = []
    for batch in tqdm(eval_dataloader):
        losses.extend(get_loss(generator, batch, labels_loss))

    losses = torch.tensor(losses, dtype=torch.float32).reshape(-1, len(labels))
    results = classify(losses, labels, p_cf, mode, return_probs=return_probs)

    return results


def load_generator(model_name, cache_dir=None, precision='fp16', local_files_only=False, device_map="auto"):
    torch.backends.cudnn.deterministic = True

    precision = torch.float16 if precision == 'fp16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        names_to_checkpoints[model_name], cache_dir=cache_dir,
        torch_dtype=precision, device_map=device_map, padding_side='right',
        trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        names_to_checkpoints[model_name], cache_dir=cache_dir,
        torch_dtype=precision, device_map=device_map,
        trust_remote_code=True, local_files_only=local_files_only)
    
    if 'llama' in model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = 29_999
        model.config.pad_token_id = 29_999
    else:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    generator = Generator(model=model, tokenizer=tokenizer)
    return generator

def generate_random_templates(all_templates, n=10, seed=59, with_big_seps=False):
    random.seed(seed)
    inp_verbalizers, out_verbalizers, seps = all_templates
    random_templates = []
    for _ in range(n):
        inp_verbalizer = random.choice(inp_verbalizers)
        out_verbalizer = random.choice(out_verbalizers)
        sep = random.choice(seps)
        big_sep = random.choice([" ", "\n", "\n\n"]) if with_big_seps else ''
        random_templates.append((inp_verbalizer, out_verbalizer, sep, big_sep))

    return random_templates
