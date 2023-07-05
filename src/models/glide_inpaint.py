from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
import torch


class GlideWrap():
    def __init__(self, model_id, guidance_scale, upsample_temp, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.has_cuda = (self.device == 'cuda')
        # Create base glide.
        self.options = model_and_diffusion_defaults()
        self.options['use_fp16'] = self.has_cuda
        self.options['inpaint'] = True
        self.options['use_fp16'] = self.has_cuda
        self.options['timestep_respacing'] = 'fast27'  # use 27 diffusion steps for very fast sampling
        self.glide, self.diffusion = create_model_and_diffusion(**self.options)
        self.glide.eval()
        if self.has_cuda:
            self.glide.convert_to_fp16()
        self.glide.to(self.device)
        self.glide.load_state_dict(load_checkpoint(model_id, self.device))

        self.guidance_scale = guidance_scale
        self.upsample_temp = upsample_temp

    def model_fn(self, x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.glide(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def __call__(self, real_images, mask_images, prompts):
        """Returns a tensor of images where the ith image is generated from the ith prompt in [prompts].

        Args:
        prompts    -- list of string prompts
        """
        batch_size = len(prompts)
        tokens = [self.glide.tokenizer.encode(p) for p in prompts]
        tokens_and_masks = [self.glide.tokenizer.padded_tokens_and_mask(t, self.options['text_ctx']) for t in tokens]
        tokens = [t for t, _ in tokens_and_masks]
        masks = [m for _, m in tokens_and_masks]

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = self.glide.tokenizer.padded_tokens_and_mask([], self.options['text_ctx'])

        # Pack the tokens together into glide kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(tokens + [uncond_tokens] * batch_size, device=self.device),
            mask=torch.tensor(masks + [uncond_mask] * batch_size, dtype=torch.bool, device=self.device),
            inpaint_image=(real_images * mask_images).repeat(2, 1, 1, 1).to(self.device),
            inpaint_mask=mask_images.repeat(2, 1, 1, 1).to(self.device)
        )

        self.glide.del_cache()
        samples = self.diffusion.p_sample_loop(
            self.model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.glide.del_cache()

        return (samples + 1) / 2
