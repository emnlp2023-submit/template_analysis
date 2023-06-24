import os
import random
from collections import defaultdict

import torch
import wandb

from utils import (
    load_generator,
    predict,
    load_split_dataset,
    parse_args,
    generate_random_templates,
    TensorDataset
)
from templates import dataset_templates


def main(dataset, models, args, seed=59, zero_shot=True, batch_size=None, methods=None, local_files_only=False,
         save_dir='results/prediction_methods', cache_dir='/mnt/data/hf_models', n_template_seeds=10):
    train, val, labels_mp = load_split_dataset(dataset, cache_dir=cache_dir)
    labels = list(labels_mp.values())

    random.seed(seed)

    os.makedirs(os.path.join(save_dir, dataset), exist_ok=True)
    res_path = os.path.join(save_dir, dataset, "res")
    if zero_shot:
        examples = []
        big_sep = ''
        res_path += '_zero_shot'
        ids = None
    else:
        ids = random.sample(range(len(train)), 2)
        examples = [tuple(v for k, v in train.iloc[idx].items()) for idx in ids]

    random_templates = generate_random_templates(dataset_templates(dataset), seed=seed, with_big_seps=not zero_shot,
                                                 n=n_template_seeds)

    model_stats = {model: defaultdict(list) for model in models}

    if not methods:
        methods = ['direct_True', 'direct_False',
                   'channel_True', 'channel_False',
                   'calibrate_True', 'calibrate_False']
    for model in models:
        generator = load_generator(model, cache_dir=cache_dir, precision=args.precision,
                                   local_files_only=args.local_files_only)
        tokenizer = generator.tokenizer

        for method_name in methods:
            print(model, method_name)
            if "_" in method_name:
                *method, labels_loss = method_name.split("_")
                method = '_'.join(method)
                labels_loss = labels_loss == 'True'
            else:
                method = method_name
                labels_loss = args.labels_loss

            wandb.init(entity=args.wandb_entity, project=args.wandb_project, reinit=True)
            config = {}
            config['prediction_method'] = method_name
            config['dataset'] = dataset
            config['n_shots'] = 0 if zero_shot else 2
            config['model'] = model
            config['example_selection_method'] = '0-shot' if zero_shot else 'random'
            config['seed'] = seed
            config['example_ids'] = ids
            config['batch_size'] = args.eval_batch_size
            config['precision'] = args.precision
            wandb.config.update(config)

            for template_seed in range(n_template_seeds):
                template = random_templates[template_seed]

                eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                             tokenizer, labels, template, examples=examples,
                                             method=method, max_length=1024)
                if 'calibrate' in method:
                    context_free_inputs = ["N/A", "", "[MASK]"]
                    calibrate_dataset = TensorDataset(context_free_inputs, tokenizer, labels, template,
                                                      examples=examples, method='direct', max_length=1024)
                else:
                    calibrate_dataset = None

                results = predict(generator, eval_dataset, labels, batch_size=batch_size, method=method,
                                  labels_loss=labels_loss, calibrate_dataset=calibrate_dataset)
                score = (results == val['target']).mean()

                model_stats[model][method_name].append(score)
                torch.save(model_stats, res_path)
            wandb.log({'templates': random_templates, 'scores': model_stats[model][method_name]})


if __name__ == "__main__":
    args = parse_args()
    zero_shot = args.num_shots == 0
    for d in args.dataset:
        main(dataset=d, models=args.inference_models, zero_shot=zero_shot, cache_dir=args.cache_dir,
             methods=args.prediction_method, batch_size=args.eval_batch_size, save_dir=args.save_dir, args=args,
             seed=args.seed[0], local_files_only=args.local_files_only
             )
