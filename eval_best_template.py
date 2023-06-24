import os
import random

import torch

from utils import load_generator, predict, load_split_dataset, parse_args, TensorDataset
    
    
best_templates = {
    'sst2': {
        'direct_False': {
           'gpt2-large': ['{}', 'sentiment: {}', ' '],
           'gpt2-xl': ['{}', 'sentiment: {}', ' '],
           'gptj': ['{}', 'All in all {}', '. '],
           'gpt-neox': ['sentence: {}', '\nAll in all {}', '.'],
           'opt-1.3b': ['sentence: {}', 'sentiment: {}', '\n'],
           'opt-6.7b': ['sentence: {}', 'All in all {}', '. '],
           'bloom-1.7b': ['text: {}', 'emotion: {}', ' '],
           'bloom-3b': ['input: {}', 'A {}', ' one. '],
           'bloom-7.1b': ['{}', 'All in all {}', '.\n'],
           'pythia-6.9b': ['{}', 'All in all {}', '.\n'],
           'pythia-12b': ['sentence: {}', 'emotion: {}', ' '],
           'cerebras-6.7b': ['sentence: {}', 'A {}', ' piece. '],
           'llama-7b': ['sentence: {}', 'It was {}', '. '],
           'llama-13b': ['{}', 'It was {}', '. '],
           'llama-30b': ['sentence: {}', 'sentiment: {}', '\n'],
           'llama-65b': ['text: {}', 'emotion: {}', ' ']
        }
    }
}


def main(args, seed):
    random.seed(seed)
    dataset = args.dataset[0]
    method = f"{args.prediction_method[0]}_{args.labels_loss}"
    
    train, val, labels_mp = load_split_dataset(dataset, cache_dir=args.cache_dir)
    labels = list(labels_mp.values())
    
    ids = random.sample(range(len(train)), args.num_shots)
    examples = [tuple(v for k, v in train.iloc[idx].items()) for idx in ids]
    
    os.makedirs(args.save_dir, exist_ok=True)

    for model in args.inference_models:
        res_path = os.path.join(args.save_dir, f"{model}_{dataset}_{method}_{args.num_shots}shot_{seed}")
        
        generator = load_generator(model, cache_dir=args.cache_dir, precision=args.precision,
                                   local_files_only=args.local_files_only)
        tokenizer = generator.tokenizer
        best_template = best_templates[dataset][method][model]

        scores = {}
        for big_sep in ["\n"]:
            template = best_template + [big_sep]
            eval_dataset = TensorDataset([x.strip() for x in val['input']],
                                            tokenizer, labels, template, examples=examples,
                                            method=method, max_length=1024)
            if 'calibrate' in method:
                context_free_inputs = ["N/A", "", "[MASK]"]
                calibrate_dataset = TensorDataset(context_free_inputs, tokenizer, labels, template=template,
                                                    examples=examples, method='direct', max_length=1024)
            else:
                calibrate_dataset = None

            results = predict(generator, eval_dataset, labels, batch_size=args.eval_batch_size, method=method,
                                labels_loss=args.labels_loss, calibrate_dataset=calibrate_dataset)
            score = (results == val['target']).mean()
            scores[big_sep] = score
        torch.save(scores, res_path)
                            

if __name__ == "__main__":
    args = parse_args()
    seeds = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seeds:
        main(args, seed)
