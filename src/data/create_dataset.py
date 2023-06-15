import json
import os
import re
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def get_metadata_dataset(metadata_file_path, source_dir):
    metadata = pd.read_json(metadata_file_path).to_dict(orient='records')
    source_imgs = os.listdir(source_dir)
    for ix, instance in tqdm(enumerate(metadata)):
        name = instance['artwork'].split('.')[0]
        references = list(filter(lambda x: x.startswith(name), source_imgs))
        crops = list(filter(lambda x: re.match(f'{name}_crop\d\d\d.png', x) ,references))
        masks = list(filter(lambda x: re.match(f'{name}_crop\d\d\d_mask\d\d\d.png', x) ,references))
        metadata[ix]['crops'] = crops
        metadata[ix]['masks'] = masks
    return metadata


def get_caption_dataset(metadata_file_path, source_dir, captions_file_path, combined=False):
    metadata = get_metadata_dataset(metadata_file_path, source_dir)
    captions = pd.read_csv(captions_file_path)
    captions.index = captions['image']
    captions.drop(['name', 'image', 'human'], axis=1)
    for ix, artwork in tqdm(enumerate(metadata)):
        metadata[ix]['caption'] = captions.loc[artwork["artwork"]].caption
    if not combined:
        return [
            {k: v for k, v in x.items() if k != 'style' and k != 'genre'}
            for x in metadata
        ]
    return metadata


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--metadatapath', type=str)
    argparser.add_argument('--indir', type=str)
    argparser.add_argument('--outname', type=str)
    argparser.add_argument('--captionspath', type=str)
    argparser.add_argument('--combined', type=bool)
    return argparser.parse_args()


def launch():
    args = parse_args()
    if args.captionspath is not None:
        metadata = get_caption_dataset(args.metadatapath, args.indir, args.captionspath, args.combined)
    else:
        metadata = get_metadata_dataset(args.metadatapath, args.indir)
    with open(args.outname, 'w+') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    launch()
