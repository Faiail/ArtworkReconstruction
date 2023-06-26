import diffusers.pipelines
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize, ToPILImage
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from torch.nn.functional import interpolate
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image


class ArtInpaint(nn.Module):
    def __init__(self, model_id, text_encoder_id):
        super().__init__()
        self.model_id = model_id
        self.text_encoder_id = text_encoder_id
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_id)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')

    def _prepare_masked_image(self, img, mask):
        h, w = [img.shape[-1]] * 2
        return prepare_mask_and_masked_image(img, mask, h, w)

    def forward(self, img, mask, text):
        # convert images to latent space
        latents = self.vae.encode(img).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        mask, masked_image = self._prepare_masked_image(img, mask)
        masked_latents = self.vae.encode(masked_image).latent_dist.sample()
        masked_latents = masked_latents * self.vae.config.scaling_factor


        # resize mask to latents shape to be passed for getting noise
        resolution = img.shape[-1]
        mask = torch.stack(
            [
                interpolate(m.unsqueeze(dim=0), size=(resolution // 8, resolution // 8)).squeeze()
                for m in mask
            ]
        )
        mask = mask.reshape(-1, 1, resolution // 8, resolution // 8)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # concatenate the noised latents with the mask and the masked latents
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(text)[0]

        # Predict the noise residual
        noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        return noise_pred, target

    @property
    def pipe(self):
        return StableDiffusionInpaintPipeline.from_pretrained(self.model_id,
                                                              unet=self.unet,
                                                              text_encoder=self.text_encoder,
                                                              scheduler=self.noise_scheduler,
                                                              vae=self.vae)

    def inpaint_image(self, image, text, mask):
        pipeline = self.pipe
        return pipeline(prompt=text, image=image, mask_image=mask,
                        num_inference_steps=10)


def transform_image(img):
    comp = Compose(
        [
            Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            CenterCrop((512, 512)),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )
    return comp(img)


def prepare_data_for_instance_test():
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    prompt = """The image represents the last supper."""
    ids = tokenizer(prompt, padding='do_not_pad',
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors='pt').input_ids
    img = Image.open('test_images/the-last-supper-1495_crop000.jpg').convert('RGB')
    mask = Image.open('test_images/the-last-supper-1495_crop000_mask000.jpg')#.convert('RGB')

    return img, mask, ids, prompt


if __name__ == '__main__':
    model = ArtInpaint('runwayml/stable-diffusion-inpainting', 'openai/clip-vit-large-patch14')
    print(model.unet)

    img, mask, ids, prompt = prepare_data_for_instance_test()
    #img.show()

    img = transform_image(img).unsqueeze(dim=0)
    mask = ToTensor()(mask).unsqueeze(dim=0)

    print(img.shape, mask.shape)
    #print(model(img, mask, ids))

    model.inpaint_image(img, prompt, mask).images[0].show()



