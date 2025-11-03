import torch

def compute_log_ratio(logprob_new, logprob_old):
    return (logprob_new - logprob_old)

def clip_ratio(r, eps=0.2):
    return torch.clamp(r, 1.0 - eps, 1.0 + eps)
