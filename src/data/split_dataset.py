import os
import warnings
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


def split_dataset(dataset_path:str, on: str = 'genre', val_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
    dataset = pd.read_json(dataset_path, orient='records')
    if on not in dataset.columns:
        warnings.warn(f'{on} not in dataset columns. Performing random splitting')
    # performing splitting
    train, drop = train_test_split(dataset,
                                   test_size=val_size+test_size,
                                   stratify=dataset[on] if on in dataset.columns else None,
                                   shuffle=True,
                                   random_state=seed)
    val, test = train_test_split(drop, test_size=test_size/(val_size+test_size),
                                 stratify=drop[on] if on in drop.columns else None,
                                 shuffle=True,
                                 random_state=seed)
    return train, val, test


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--dataset-path')
    argument_parser.add_argument('--val-size', default=0.1)
    argument_parser.add_argument('--test-size', default=0.1)
    argument_parser.add_argument('--target-strat', default='genre')
    argument_parser.add_argument('--seed', default=42)
    argument_parser.add_argument('--outdir', default='splitted_data')
    return argument_parser.parse_args()


def launch():
    args = parse_args()
    train, val, test = split_dataset(dataset_path=args.dataset_path,
                                     on=args.target_strat,
                                     val_size=args.val_size,
                                     test_size=args.test_size,
                                     seed=args.seed)
    train = train.to_dict(orient='records')
    val = val.to_dict(orient='records')
    test = test.to_dict(orient='records')

    os.makedirs(args.outdir, exist_ok=True)

    with open(f'{args.outdir}/train.json', 'w+') as f:
        json.dump(train, f)

    with open(f'{args.outdir}/val.json', 'w+') as f:
        json.dump(val, f)

    with open(f'{args.outdir}/test.json', 'w+') as f:
        json.dump(test, f)


if __name__ == '__main__':
    launch()

