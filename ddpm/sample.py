import torch

def p_sample_ddim(model, x_t, t_cur, t_prev, alphas_cumprod, eta=0.0):
    alpha_bar_t = alphas_cumprod[t_cur - 1]
    if t_prev > 0:
        alpha_bar_prev = alphas_cumprod[t_prev - 1]
    else:
        alpha_bar_prev = torch.tensor(1.0, device=x_t.device)
    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
    B = x_t.size(0)
    t_tensor = torch.full((B,), t_cur, device=x_t.device, dtype=torch.long)
    eps_theta = model(x_t, t_tensor)
    sqrt_ab_t = torch.sqrt(alpha_bar_t)
    sqrt_ab_prev = torch.sqrt(alpha_bar_prev)
    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1) * eps_theta) / sqrt_ab_t.view(-1, 1, 1, 1)
    dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0.0)).view(-1, 1, 1, 1) * eps_theta
    noise = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)
    x_prev = sqrt_ab_prev.view(-1, 1, 1, 1) * x0_pred + dir_xt + sigma_t.view(-1, 1, 1, 1) * noise
    return x_prev, eps_theta.detach().cpu(), x_t.detach().cpu()

def sample_ddim(model, shape, alphas_cumprod, device, ddim_steps, eta=0.0):
    steps_log = {"t": [], "eps_t": [], "x_t": []}
    num_timesteps = alphas_cumprod.shape[0]
    x = torch.randn(shape, device=device)
    idx_lin = torch.linspace(0, num_timesteps - 1, steps=ddim_steps + 1, device=device)
    idx0 = idx_lin.round().long()
    idx0 = torch.cat([torch.tensor([0, num_timesteps - 1], device=device, dtype=torch.long), idx0]).unique(sorted=True)
    seq_asc = idx0 + 1
    seq_rev = torch.flip(seq_asc, dims=[0])
    seq = torch.cat([seq_rev, torch.tensor([0], device=device, dtype=torch.long)])
    prev_t = seq[0].item()
    for next_t in seq[1:]:
        t_cur = prev_t
        t_prev = next_t.item()
        x, eps_t, x_t_snap = p_sample_ddim(
            model,
            x,
            t_cur,
            t_prev,
            alphas_cumprod,
            eta
        )
        steps_log["t"].append(t_cur)
        steps_log["eps_t"].append(eps_t)
        steps_log["x_t"].append(x_t_snap)
        prev_t = t_prev
    return x, steps_log