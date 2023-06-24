import os

import torch
import random

from utils import (
    load_generator,
    load_split_dataset,
    predict,
    names_to_checkpoints,
    parse_args,
    TensorDataset,
)


def dataset_templates(dataset):
    inp_verbalizers = ['input: {}', 'text: {}', 'sentence: {}', '{}']
    out_verbalizers = ['output: {}', 'target: {}', 'label: {}']
    seps = ['\n', ' ']

    if dataset == 'sst2':
        out_verbalizers += ['emotion: {}', 'sentiment: {}', 'A {} one.', 'It was {}.', 'All in all {}.', 'A {} piece.']
    elif dataset in ['dbpedia', 'agnews', 'trec']:
        out_verbalizers += ['Topic: {}.', 'Subject: {}.', 'This is about {}.', 'It is about {}.']
    else:
        raise NotImplementedError

    out_verbalizers.append('{}')

    return inp_verbalizers, out_verbalizers, seps


def main(dataset, args, method, labels_loss, zero_shot=True, cache_dir=None):
    if isinstance(dataset, list):
        dataset = dataset[0]
    train, val, labels_mp = load_split_dataset(dataset)
    labels = list(labels_mp.values())

    inp_verbalizers, out_verbalizers, seps = dataset_templates(dataset)

    inference_models = args.inference_models if args.inference_models else names_to_checkpoints
    print("Running on the following models:", inference_models)
    res_dir = args.save_dir if args.save_dir else f"results/template_selection/{dataset}"
    os.makedirs(res_dir, exist_ok=True)
    print("Results will be saved to", res_dir)
    if isinstance(inference_models, str):
        inference_models = [inference_models]

    for inference_model in inference_models:
        torch.cuda.empty_cache()
        generator = load_generator(inference_model, cache_dir=cache_dir, local_files_only=args.local_files_only, device_map=args.device_map)
        tokenizer = generator.tokenizer

        stats = {}

        if zero_shot:
            examples = []
            big_seps = ['']
            ids = None
        else:
            ids = random.sample(range(len(train)), args.num_shots)
            examples = [tuple(v for k, v in train.iloc[idx].items()) for idx in ids]
            big_seps = [' ', '\n']

        res_file = os.path.join(res_dir, f"{inference_model}_formats_stats")
        if zero_shot:
            res_file += '_zero_shot'
        res_file += f'_{method[0]}_{labels_loss}'
        
        if os.path.exists(res_file):
            print("Found results for", inference_model)
            stats = torch.load(res_file)

        for inp_verbalizer in inp_verbalizers:
            for out_verbalizer in out_verbalizers:
                for sep in seps:
                    for big_sep in big_seps:
                        template = inp_verbalizer, out_verbalizer, sep, big_sep
                        if ''.join(template) in stats:
                            print(f"{template} skipped")
                            continue

                        context = ""

                        if method == 'channel':
                            context += big_sep.join([f"{out_verbalizer.format(x[1])}{sep}{inp_verbalizer.format(x[0])}"
                                                    for x in examples])
                        else:
                            context += big_sep.join([f"{inp_verbalizer.format(x[0])}{sep}{out_verbalizer.format(x[1])}"
                                                    for x in examples])
                        if context:
                            context += big_sep

                        if 'calibrate' in method:
                            context_free_inputs = ["N/A", "", "[MASK]"]
                            calibrate_dataset = TensorDataset(context_free_inputs, tokenizer, labels, template,
                                                              examples=examples, method='direct', max_length=1024)
                        else:
                            calibrate_dataset = None

                        eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                     tokenizer, labels, template, examples=examples,
                                     method=method, max_length=1024)

                        results = predict(generator, eval_dataset, labels, batch_size=args.eval_batch_size, method=method,
                                          labels_loss=labels_loss, calibrate_dataset=calibrate_dataset, return_probs=False)

                        score = (results == val['target'].values).mean()
                        stats[big_sep + ''.join(template)] = score

                        torch.save(stats, res_file)


if __name__ == '__main__':
    args = parse_args()
    zero_shot = args.num_shots == 0
    main(args.dataset, args, method=args.prediction_method, labels_loss=args.labels_loss, cache_dir=args.cache_dir, zero_shot=zero_shot)
