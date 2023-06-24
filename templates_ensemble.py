import os
import random
from collections import defaultdict

import torch

from utils import (
    load_generator,
    predict,
    load_split_dataset,
    parse_args,
    generate_random_templates,
    TensorDataset
)
from templates import dataset_templates


def main(dataset, models, args, seed=59, zero_shot=True, batch_size=None, methods=None,
         save_dir='templates_ensemble', cache_dir='/mnt/data/hf_models', n_template_seeds=5):
    train, val, labels_mp = load_split_dataset(dataset, seed=59, cache_dir=cache_dir)
    labels = list(labels_mp.values())

    seeds = range(n_template_seeds)
    random.seed(seed)

    res_path = os.path.join(save_dir, dataset)
    if zero_shot:
        examples = []
        res_path = os.path.join(res_path, "zero_shot")
        ids = None
    else:
        ids = random.sample(range(len(train)), args.num_shots)
        examples = [tuple(v for k, v in train.iloc[idx].items()) for idx in ids]
        res_path = os.path.join(res_path, f"{args.num_shots}_shot")
    
    os.makedirs(res_path, exist_ok=True)

    random_templates = generate_random_templates(dataset_templates(dataset), seed=seed, with_big_seps=not zero_shot,
                                                 n=n_template_seeds)

    model_stats = {model: defaultdict(dict) for model in models}

    for model in models:
        generator = load_generator(model, cache_dir=cache_dir, precision=args.precision, device_map=args.device_map, local_files_only=args.local_files_only)
        tokenizer = generator.tokenizer

        if isinstance(methods, str):
            methods = [methods]

        for method_name in methods:
            model_stats[model][method_name]["score"] = []
            model_stats[model][method_name]["results"] = []
            model_stats[model][method_name]["probs"] = []

            print(model, method_name)
            if "_" in method_name:
                *method, labels_loss = method_name.split("_")
                method = '_'.join(method)
                labels_loss = labels_loss == 'True'
            else:
                method = method_name
                labels_loss = args.labels_loss

            for template_seed in range(len(seeds)):
                template = random_templates[template_seed]

                print(examples)
                eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                             tokenizer, labels, template, examples=examples,
                                             method=method, max_length=1024)
                if 'calibrate' in method:
                    context_free_inputs = ["N/A", "", "[MASK]"]
                    calibrate_dataset = TensorDataset(context_free_inputs, tokenizer, labels, template,
                                                      examples=examples, method='direct', max_length=1024)
                else:
                    calibrate_dataset = None

                results, probs = predict(generator, eval_dataset, labels, batch_size=batch_size, method=method,
                                  labels_loss=labels_loss, calibrate_dataset=calibrate_dataset, return_probs=True,
                                skip_cuda_oom=args.skip_cuda_oom)
                score = (results == val['target']).mean()

                model_stats[model][method_name]["score"].append(score)
                model_stats[model][method_name]["results"].append(results)
                model_stats[model][method_name]["probs"].append(probs)
                torch.save(model_stats[model][method_name], os.path.join(res_path, f"{model}_{method}_{seed}"))

if __name__ == "__main__":
    args = parse_args()
    zero_shot = args.num_shots == 0
    for d in args.dataset:
        for seed in args.seed:
            main(dataset=d, models=args.inference_models, zero_shot=zero_shot, cache_dir=args.cache_dir,
                     methods=args.prediction_method, batch_size=args.eval_batch_size, save_dir=args.save_dir, args=args,
                     seed=seed, n_template_seeds=args.num_templates
                )
