from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from tqdm import tqdm



def main():
    pipe = StableDiffusionInpaintPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    print(pipe)
    vae = pipe.vae
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    pipe2 = StableDiffusionInpaintPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',vae = vae,
                                                           scheduler=scheduler,
                                                           text_encoder=text_encoder,
                                                           unet=unet)

if __name__ == '__main__':
    main()
    exit()
    device = 'cpu'
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4',
                                        subfolder='vae')
    vae = vae.to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)

    img = Image.open('./test_images/the-last-supper-1495_crop000.jpg').convert('RGB')
    mask = Image.open('./test_images/the-last-supper-1495_crop000_mask000.jpg')
    out = prepare_mask_and_masked_image(img, mask, 512, 512, True)
    mask, masked, img = out
    print(masked.shape, mask.shape, img.shape)
    exit()
    #img.show()
    img = ToTensor()(img)
    scheduler.set_timesteps(2)
    #scheduler.add_noise(img, torch.randn(img.shape), scheduler.timesteps)

    prompt = 'the last supper.'
    text_input = tokenizer(prompt, padding='max_length', return_tensors="pt")
    uncond_input = tokenizer('', padding = 'max_length', return_tensors='pt')

    text_embeddings = text_encoder(text_input.input_ids)[0]
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


    latents = vae.encode(img.unsqueeze(dim=0)).latent_dist.sample()
    print(latents.shape)

    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    #image = (image / 2 + 0.5).clamp(0, 1)
    print(image.shape)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    print(ToTensor()(pil_images[0]).shape)

    pil_images[0].show()






