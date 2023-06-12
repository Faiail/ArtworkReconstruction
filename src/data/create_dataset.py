import json
import os
import re
from tqdm import tqdm
from argparse import ArgumentParser

#TODO: AGGIUNGI IL CASO IN CUI VUOI CRARE IL DATASET CON LE CAPTIONS


def get_metadata_dataset(metadata_file_path, source_dir):
    with open(metadata_file_path) as f:
        metadata = json.load(f)
    source_imgs = os.listdir(source_dir)
    for ix, instance in tqdm(enumerate(metadata[:10])):
        name = instance['artwork'].split('.')[0]
        references = list(filter(lambda x: x.startswith(name), source_imgs))
        crops = list(filter(lambda x: re.match(f'{name}_crop\d\d\d.png', x) ,references))
        masks = list(filter(lambda x: re.match(f'{name}_crop\d\d\d_mask\d\d\d.png', x) ,references))
        metadata[ix]['crops'] = crops
        metadata[ix]['masks'] = masks
    return metadata


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--metadatapath', type=str)
    argparser.add_argument('--indir', type=str)
    argparser.add_argument('--outname', type=str)
    return argparser.parse_args()


def launch():
    args = parse_args()
    metadata = get_metadata_dataset(args.metadatapath, args.indir)
    with open(args.outname, 'w+') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    launch()