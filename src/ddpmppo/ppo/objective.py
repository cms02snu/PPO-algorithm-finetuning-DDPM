import torch

def ppo_clip_objective(ratio, adv, eps=0.2):
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
    loss = -torch.mean(torch.minimum(unclipped, clipped))  # gradient descent
    return loss
