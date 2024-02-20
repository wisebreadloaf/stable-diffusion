import torch
import numpy as np


class DOPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000,
        beta_Start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_Start**0.5, beta**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_nump(np.arange(0, num_training_steps)[::-1]).copy()

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)
