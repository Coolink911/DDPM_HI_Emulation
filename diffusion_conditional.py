"""
Diffusion Process Implementation (DDPM + DDIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion:
    def __init__(self, timesteps=1500, beta_start=1e-4, beta_end=0.02, schedule_type="linear"):
        self.timesteps = timesteps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, model, x_t, t, labels, clip_denoised=True):
        pred_noise = model(x_t, t, labels)
        x_start = self._predict_xstart_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance, _ = self.q_posterior_mean_variance(x_start, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _predict_xstart_from_noise(self, x_t, t, noise):
        return (
            self._extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_sample(self, model, x_t, t, labels):
        model_mean, _, model_log_variance, _ = self.p_mean_variance(model, x_t, t, labels)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def ddim_sample_step(self, model, x_t, t, t_next, labels, eta=0.0):
        pred_noise = model(x_t, t, labels)
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t_next = self._extract(self.alphas_cumprod, t_next, x_t.shape) if t_next[0] >= 0 else torch.ones_like(alpha_t)

        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t)) * pred_noise

        noise = torch.randn_like(x_t) if eta > 0 else 0
        sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)) if eta > 0 else 0

        return torch.sqrt(alpha_t_next) * x0_pred + dir_xt + sigma_t * noise

    def sample(self, model, labels, channels, height, width, device, progress=False, use_ddim=True, ddim_steps=50, eta=0.0):
        batch_size = labels.shape[0]
        img = torch.randn((batch_size, channels, height, width), device=device)

        if use_ddim:
            skip = self.timesteps // ddim_steps
            seq = list(range(0, self.timesteps, skip))
            seq_next = [-1] + seq[:-1]
            seq_iter = reversed(list(zip(seq, seq_next)))
            if progress:
                from tqdm import tqdm
                seq_iter = tqdm(seq_iter, desc=f'DDIM Sampling ({ddim_steps} steps)', total=len(seq))
            for i, j in seq_iter:
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                t_next = torch.full((batch_size,), j, device=device, dtype=torch.long)
                img = self.ddim_sample_step(model, img, t, t_next, labels, eta)
        else:
            # DDPM sampling (slow)
            if progress:
                from tqdm import tqdm
                timesteps_iter = tqdm(reversed(range(self.timesteps)), total=self.timesteps)
            else:
                timesteps_iter = reversed(range(self.timesteps))
            for i in timesteps_iter:
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                img = self.p_sample(model, img, t, labels)
        return img

    def training_losses(self, model, x_start, labels, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(x_t, t, labels)
        return F.mse_loss(pred_noise, noise, reduction='none').mean(dim=list(range(1, len(pred_noise.shape))))

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def to(self, device):
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                     'posterior_variance', 'posterior_log_variance_clipped',
                     'posterior_mean_coef1', 'posterior_mean_coef2']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, unet, diffusion_process):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion_process

    def forward(self, x, t, labels):
        return self.unet(x, t, labels)

    def get_loss(self, x, labels, noise=None):
        batch_size = x.shape[0]
        device = x.device
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device).long()
        return self.diffusion.training_losses(self, x, labels, t, noise=noise).mean()

    def sample(self, labels, channels, height, width, device, progress=False, use_ddim=True, ddim_steps=50, eta=0.0):
        self.eval()
        with torch.no_grad():
            return self.diffusion.sample(self, labels, channels, height, width, device, progress, use_ddim, ddim_steps, eta)
