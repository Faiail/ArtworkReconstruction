import torch
from torch import nn
from torch.nn.functional import gelu


def pos_encoding(t, channels, device='cuda'):
    t = t.cpu()
    inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc.to(device)


class CrossModalMultiHeadAttention(torch.nn.Module):
    def __init__(self, img_size, channels, text_size, hidden_state):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.text_size = text_size
        self.txt_proj = nn.Linear(text_size, hidden_state)
        self.img_proj = nn.Linear(img_size*img_size, hidden_state)
        self.mha = torch.nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, img, text):
        img = img.view(-1, self.channels, self.img_size * self.img_size)
        img = self.img_proj(img).swapaxes(1, 2)
        text = text.view(-1, 1, self.text_size).repeat(1, self.channels, 1)
        text = self.txt_proj(text).swapaxes(1, 2)
        attention_value, _ = self.mha(img, text, text)
        attention_value = attention_value + img
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.img_size, self.img_size)


class DownConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, residual=False):
        super().__init__()
        self.residual = residual
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x) if not self.residual else gelu(x + self.maxpool_conv(x))
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, residual=True, emb_dim=256):
        self.residual = residual
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x) if not self.residual else gelu(x + self.conv(x))
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class ImageEncoder(torch.nn.Module):
    def __init__(self, text_size=768, time_emb=256, img_size=512):
        super().__init__()
        self.conv0 = DownConv(in_channels=3,
                              out_channels=64,
                              emb_dim=time_emb,
                              residual=False)
        self.att0 = CrossModalMultiHeadAttention(img_size=img_size//2, channels=64, text_size=text_size)
        self.downscale0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = DownConv(in_channels=64,
                              out_channels=128,
                              emb_dim=time_emb,
                              residual=False)
        self.downscale1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = DownConv(in_channels=128,
                              out_channels=256,
                              emb_dim=time_emb,
                              residual=False)
        self.downscale2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, t, text):
        t = pos_encoding(t, 256)
        x = self.conv0(x, t)
        x = self.att0(x, text)
        x = self.downscale0(x)
        x = self.conv1(x, t)
        x = self.downscale1(x)
        x = self.conv2(x, t)
        x = self.downscale2(x)
        return x

class ImageDecoder(torch.nn.Module):
    def __init__(self):
        pass



class Diffusion:
    def __init__(self, noise_steps=1e3, beta_start=1e-4, beta_end=0.02, img_size=512, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


if __name__ == '__main__':
    enc = ImageEncoder(img_size=512).to('cuda')
    x = torch.randn(2, 3, 512, 512).to('cuda')
    text = torch.randn(2, 768).to('cuda')
    print(enc(x, torch.Tensor([5]).to('cuda'), text).shape)