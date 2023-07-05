import torch.cuda
import sys
sys.path.append('../')
from models.glide_inpaint import GlideWrap
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    InterpolationMode
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader
from eval_dataset import EvalDataset
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import json


def load_dataset(root_dir, dataset_path, bs=16, seed=42, size=64):
    transform = Compose(
        [
            ToTensor(),
            Resize(size, interpolation=InterpolationMode.BICUBIC),
        ]
    )
    data = pd.read_json(dataset_path)
    dataset = EvalDataset(root_dir=root_dir,
                          data=data,
                          transform=transform,
                          seed=seed)
    return DataLoader(dataset, shuffle=True, batch_size=bs, drop_last=False)


def test(test_loader, model):
    fid = FrechetInceptionDistance(normalize=True)
    is_ = InceptionScore(normalize=True)
    for batch in tqdm(test_loader):
        out = model(batch['crop'], batch['mask'], batch['prompt'])
        fid.update(out.cpu(), real=False)
        fid.update(batch['crop'], real=True)
        is_.update(out.cpu())
    mean, std = is_.compute()
    fid_val = fid.compute().item()
    return mean.item(), std.item(), fid_val


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--model-id', help='name of the pretrained pipeline.',
                                 default="base-inpaint")
    argument_parser.add_argument('--root-dir', help='folder in which the images are stored.',
                                 required=True)
    argument_parser.add_argument('--dataset-path', help='json file in which are annotated all the captions.',
                                 required=True)
    argument_parser.add_argument('--batch-size', default=16)
    argument_parser.add_argument('--guidance', default=3.0)
    argument_parser.add_argument('--upsample-temp', default=0.997)
    argument_parser.add_argument('--result-path', help='path to output json file.', required=True)
    return argument_parser.parse_args()


def main(args):
    model = GlideWrap(model_id=args.model_id,
                     guidance_scale=args.guidance,
                     upsample_temp=args.upsample_temp,
                     device='cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = load_dataset(root_dir=args.root_dir,
                               dataset_path=args.dataset_path,
                               bs=args.batch_size)
    (m, s, fid) = test(test_loader, model)
    results = {
        'is': {
            'mean': m,
            'std': s,
        },
        'fid': fid
    }
    with open(args.result_path, 'w+') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main(parse_args())