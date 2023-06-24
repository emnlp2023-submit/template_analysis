import os
import json
import random
from collections import defaultdict

import torch
import wandb

from templates import dataset_templates
from utils import (
    load_generator,
    predict,
    load_split_dataset,
    parse_args,
    generate_random_templates,
    TensorDataset,
    names_to_checkpoints
)


def main(args, dataset):
    if args.inference_models is None:
        args.inference_models = list(names_to_checkpoints.keys())
    selection_method = args.examples_selection_method[0]
    examples_for_each_input = selection_method == 'z-ICL'

    train, val, labels_mp = load_split_dataset(dataset)
    labels = list(labels_mp.values())

    out_path = os.path.join(args.save_dir, dataset)
    os.makedirs(out_path, exist_ok=True)
    res_path = os.path.join(out_path, f"{selection_method}_res")
    model_stats = {model: defaultdict(list) for model in args.inference_models}

    for model in args.inference_models:
        generator = load_generator(model, cache_dir=args.cache_dir, precision=args.precision,
                                   local_files_only=args.local_files_only)
        tokenizer = generator.tokenizer

        for seed in args.seed:
            if args.example_ids is None and args.examples_path is None:
                if selection_method not in ['random', '0-shot']:
                    try:
                        examples, example_ids = load_examples(f"selected_examples/{selection_method}/{dataset}.json",
                                                 train, seed, args.num_shots)
                    except FileNotFoundError:
                        raise ValueError("Attempted to find examples in the default path. No Luck. "
                                         "All methods except zero-shot and random require either example ids or a path to the {dataset}.json")
                elif selection_method == 'random':
                    assert args.num_shots != 0
                    print(f"selecting {args.num_shots} random examples with seed {seed}")
                    random.seed(args.seed)
                    example_ids = random.sample(range(len(train)), args.num_shots)
                else:
                    # selection_method == '0-shot'
                    example_ids = []

            elif args.example_ids is not None:
                example_ids = args.example_ids
                examples = [(train['input'][idx], train['target'][idx]) for idx in example_ids]
            elif args.examples_path is not None:
                examples, example_ids = load_examples(args.examples_path, train, seed, args.num_shots)
            print("Loaded examples, first test-input Context:\n\n")
            if example_ids is None:
                print(examples[0])
            else:
                print(examples)

            random_templates = generate_random_templates(dataset_templates(dataset), seed=seed,
                                                         with_big_seps=args.num_shots > 0, n=args.num_templates)

            for prediction_method in args.prediction_method:
                method_name = f"{selection_method}-{args.num_shots}shot-{seed}-{prediction_method}"
                print(method_name)

                evaluated_templates = len(model_stats[model][method_name])
                if evaluated_templates >= args.num_templates:
                    continue

                wandb.init(entity=args.wandb_entity, project=args.wandb_project, reinit=True)
                config = {}
                config['prediction_method'] = prediction_method
                config['dataset'] = dataset
                if example_ids is None:
                    config['n_shots'] = len(examples[0])
                else:
                    config['n_shots'] = len(example_ids)
                    config['example_ids'] = example_ids
                config['model'] = model
                config['example_selection_method'] = selection_method
                config['seed'] = seed
                config['batch_size'] = args.eval_batch_size
                config['precision'] = args.precision
                wandb.config.update(config)
                for template_seed in range(evaluated_templates, len(random_templates)):
                    template = random_templates[template_seed]

                    eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                                 tokenizer, labels, template, examples=examples, 
                                                 examples_for_each_input=examples_for_each_input,
                                                 method=prediction_method, max_length=1024)
                    if 'calibrate' in prediction_method:
                        context_free_inputs = ["N/A", "", "[MASK]"]
                        calibrate_dataset = TensorDataset(context_free_inputs, tokenizer, labels, template,
                                                          examples_for_each_input=examples_for_each_input,
                                                          examples=examples, method='direct', max_length=1024)
                    else:
                        calibrate_dataset = None

                    labels_loss = 'True' in prediction_method
                    results = predict(generator, eval_dataset, labels, batch_size=args.eval_batch_size,
                                      method=prediction_method,
                                      labels_loss=labels_loss, calibrate_dataset=calibrate_dataset)
                    score = (results == val['target']).mean()

                    model_stats[model][method_name].append(score)
                    torch.save(model_stats, res_path)
                wandb.log({'templates': random_templates, 'scores': model_stats[model][method_name]})


def load_examples(path, df, seed, num_shots):
    selected_examples = json.load(open(path))
    example_ids = selected_examples[str(seed)][str(num_shots)]
    if isinstance(example_ids[0], list):
        # z-ICL examples format
        examples = [[(example_ids[i][j][0], example_ids[i][j][1]) for j in range(num_shots)] for i in range(len(example_ids))]
        example_ids = None
    else:
        assert isinstance(example_ids[0], int)
        # other methods: examples are stored as indexes in the respective dataset
        examples = [(df['input'][idx].strip(), df['target'][idx].strip()) for idx in example_ids]
    return examples, example_ids


if __name__ == '__main__':
    args = parse_args()
    for d in args.dataset:
        main(args, dataset=d)
