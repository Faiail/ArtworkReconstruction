from argparse import ArgumentParser
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from diffusers import StableDiffusionInpaintPipeline
from eval_dataset import EvalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


def load_pipeline(model_id, checker='dummy', device='cuda'):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id).to(device)
    if checker == 'dummy':
        def dummy(images, **kwargs):
            return images, [False] * images.size(0)
        pipe.safety_checker = dummy
    elif checker != 'default':
        raise ValueError(f'Invalid vale for checker parameter. Value found: {checker}, but acceptable valuer are ['
                         f'dummy, default]')
    return pipe


def load_dataset(root_dir, dataset_path, bs=4, seed=42):
    transform = Compose([ToTensor()])
    data = pd.read_json(dataset_path)#.iloc[:4]
    dataset = EvalDataset(root_dir=root_dir,
                          data=data,
                          transform=transform,
                          seed=seed)
    return DataLoader(dataset, shuffle=True, batch_size=bs, drop_last=False)


def test(test_loader, pipeline, seed=42):
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator('cuda').manual_seed(seed)
    fid = FrechetInceptionDistance(normalize=True)
    is_ = InceptionScore(normalize=True)
    for batch in tqdm(test_loader):
        out = pipeline(prompt=batch['prompt'],
                       image=batch['crop'],
                       mask_image=batch['mask'],
                       output_type='np.array',
                       generator=generator).images
        out = torch.from_numpy(out).permute(0, 3, 2, 1)
        fid.update(out, real=False)
        fid.update(batch['crop'], real=True)
        is_.update(out)
    mean, std = is_.compute()
    fid_val = fid.compute().item()
    return mean.item(), std.item(), fid_val


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--pipe-name', help='name of the pretrained pipeline.',
                                 default="runwayml/stable-diffusion-inpainting")
    argument_parser.add_argument('--root-dir', help='folder in which the images are stored.',
                                 required=True)
    argument_parser.add_argument('--dataset-path', help='csv file in which are annotated all the captions.',
                                 required=True)
    argument_parser.add_argument('--batch-size', default=4)
    argument_parser.add_argument('result-path', help='path to output json file.', required=True)
    return argument_parser.parse_args()


def main(args):
    pipe = load_pipeline(model_id=args.pipe_name, device='cuda')
    test_loader = load_dataset(root_dir=args.root_dir,
                               dataset_path=args.dataset_path,
                               bs=args.batch_size)
    (m, s, fid) = test(test_loader, pipe)
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