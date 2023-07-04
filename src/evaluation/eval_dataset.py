import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage
import random
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

class EvalDataset(Dataset):
    def __init__(self, root_dir, data, transform=None, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.data = data
        self.transform = transform
        if seed:
            random.seed(seed)

    def __getitem__(self, item):
        raw = self.data.iloc[item]
        crop = raw.crop
        mask = random.choice(raw.masks)
        prompt = raw.caption

        crop = Image.open(f'{self.root_dir}/{crop}').convert('RGB')
        crop = self.transform(crop)
        mask = Image.open(f'{self.root_dir}/{mask}')#.convert('RGB')
        mask = self.transform(mask)
        name = raw.artwork

        example = {
            'name': name,
            'crop': crop,
            'mask': mask,
            'prompt': prompt,
        }
        return example

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = pd.read_json('../data/splitted_data/test.json')
    transform = Compose(
        [ToTensor()]
    )
    dataset = EvalDataset(root_dir='/ext/raffaele/images_inpainting',
                          data=data,
                          transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    example = next(iter(loader))
    print(example['prompt'])
    print(example['crop'].shape)
    print(example['mask'].shape)

    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to('cuda')
    def dummy(images, **kwargs):
        return images, [False] * images.size(0)
    pipe.safety_checker=dummy
    out = pipe(prompt=example['prompt'], image=example['crop'], mask_image=example['mask'])
    images = out.images
    for real, rec in zip(example['crop'], images):
        ToPILImage()(real).show(title=example['name'])
        rec.show(title=f'{example["name"]} RECONSTRUCTED')
        input()