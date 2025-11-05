import torch
import numpy as np

def get_beta_alpha_linear(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    betas = torch.tensor(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return betas, alphas, alphas_cumprod

def get_alpha_bar(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    _,_,alpha_bar = get_beta_alpha_linear(beta_start,beta_end,num_timesteps)

    return alpha_bar