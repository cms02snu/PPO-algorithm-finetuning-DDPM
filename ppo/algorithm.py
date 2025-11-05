import math
import torch
from ddpm.sample import sample_ddim

def compute_mu_from_eps(x_t, t_idx, eps_pred, alphas, alpha_bars):
    """
    Compute mu(mean) from epsilon which is predicted by model.
    The value of mu is used to compute the ratio of the value of policy

    Parameters:
        x_t (torch.Tensor) : Predicted image tensor at timestep t during the sampling process.
        t_idx (int) : Number of the timestep.
        eps_pred (torch.Tensor) : The value of epsilon predicted by model.
        alphas (torch.Tensor) : Tensor of alphas.
        alpha_bars (torch.Tensor) : Tensor of alpha_bars.

    Returns:
        torch.Tensor : mu(mean)
    """
    T = alphas.shape[0]
    if t_idx < 0:
        t_idx = 0
    if t_idx > T - 1:
        t_idx = T - 1

    alpha_t = alphas[t_idx].to(x_t.device, dtype=torch.float32).view(-1, 1, 1, 1)
    beta_t  = (1.0 - alphas[t_idx].to(x_t.device, dtype=torch.float32)).view(-1, 1, 1, 1)
    abar_t  = alpha_bars[t_idx].to(x_t.device, dtype=torch.float32).view(-1, 1, 1, 1)
    denom = torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12))
    num   = beta_t * eps_pred.to(torch.float32)

    mu = (x_t.to(torch.float32) - (num / denom)) / torch.sqrt(torch.clamp(alpha_t, min=1e-12))

    return mu.to(x_t.dtype)

def compute_log_r_sum(cur_model, steps_log, x0_cpu,
                      alphas_cumprod, device, eta, eps_to_mu_fn,
                      per_step_clip=5.0, global_clip=10.0, sigma_floor=1e-3):
    """
    Compute the log of the sum of the r.
    r is the value of the ratio of the policy probability function, as in the PPO algorithm.
    This function not only computes log_r_sum, but also the gradients of each parameters.

    Parameters:
        cur_model (torch.nn.Module) : The model updated most recently.
        steps_log (dictionary) : Dictionary of t, x_t, epsilon of each timestep while sampling.
        x0_cpu (torch.Tensor) : Image tensor computed by reverse(sampling) process.
        alphas_cumprod (torch.Tensor) : Tensor of alpha_bars.
        device (torch.device) : Device.
        eta (float) : The ratio of probabilistic process in the DDPM(DDIM) reverse process.
        eps_to_mu_fn (function) : Function to compute mu(mean) from epsilon.
        per_step_clip (float) : Clipping value of log r_t in each timestep.
        global_clip (float) : Clipping value of log_r_sum.
        simga_floor (float) : Minimum clipping value of sigma to avoid sigma to be too small.

    Returns:
        float : log_r_sum. Summation is done in each timestep.
        list of tensors : sum_grads. List of gradients of each parameters. Summation is done in each timestep.
    """
    params = [p for p in cur_model.parameters() if p.requires_grad]
    sum_grads = [torch.zeros_like(p) for p in params]
    log_r_sum = 0.0

    Tlist = steps_log["t"]
    Xlist = steps_log["x_t"]
    Elist = steps_log["eps_t"]

    total_steps = len(Tlist)

    # Iter each step and compute r in each step.
    for k in range(total_steps):
        t_cur  = int(Tlist[k])
        t_prev = int(Tlist[k+1]) if (k+1) < total_steps else 0

        # Skip deterministic last-step
        if t_prev == 0:
            continue

        # Load data on GPU
        x_t = Xlist[k].to(device).detach()
        x_tprev = Xlist[k+1].to(device).detach() if (k+1)<len(Xlist) else x0_cpu.to(device).detach()
        eps_old = Elist[k].to(device).detach()

        # Compute mu_old. mu_old is computed by eps_old which is computed by old_model.
        mu_old = eps_to_mu_fn(x_t, t_cur-1, eps_old).detach()

        # Compute mu_cur. mu_cur is computed by eps_cur which is computed by cur_model.
        t_tensor = torch.full((x_t.size(0),), t_cur, device=device, dtype=torch.long)
        eps_cur  = cur_model(x_t, t_tensor)  # requires_grad=True
        mu_cur = eps_to_mu_fn(x_t, t_cur-1, eps_cur)

        # Compute sigma_t to use in DDIM sampling steps.
        a_t = alphas_cumprod[t_cur-1].to(device).float()
        if t_prev > 0:
            a_prev = alphas_cumprod[t_prev-1].to(device).float()
            num = torch.clamp(1.0-a_prev, min=0.0)
            denom = torch.clamp(1.0-a_t, min=1e-12)
            frac = torch.clamp(num/denom,  min=0.0, max=1.0)
            term = torch.clamp(1.0 - (a_t/a_prev), min=0.0)
            c1 = torch.sqrt(frac)
            c2 = torch.sqrt(term)
        else:
            c1 = torch.tensor(1.0, device=device, dtype=torch.float32)
            c2 = torch.tensor(0.0, device=device, dtype=torch.float32)
        sigma_t = (eta * c1 * c2).to(x_t.dtype)
        sigma_t = torch.clamp(sigma_t, min=float(sigma_floor))

        # Compute log r_t.
        diff_cur = (x_tprev - mu_cur).pow(2).mean() # Use mean so that the scale of the value is preserved.
        diff_old = (x_tprev - mu_old).pow(2).mean()
        denom = 2.0 * (sigma_t**2) + 1e-12
        log_r_t  = - (diff_cur-diff_old) / denom

        # Clamp before autodiff to avoid exploding grads.
        log_r_t_clamped = torch.clamp(log_r_t, -per_step_clip, per_step_clip)

        # Compute grads on clamped log value.
        grads_t = torch.autograd.grad(log_r_t_clamped, params, retain_graph=False, allow_unused=True)

        # Accumulate detached grads.
        for j, g in enumerate(grads_t):
            if (g is not None) and torch.isfinite(g).all().item():
                sum_grads[j].add_(g.detach())

        # Numeric accumulation (sum of clamped logs).
        log_r_sum += float(log_r_t_clamped.detach().cpu().item())

        # Cleanup local tensors.
        del x_t, x_tprev, eps_old, mu_old, eps_cur, mu_cur, diff_cur, diff_old, log_r_t, log_r_t_clamped, grads_t

    # Clamp computed log_r_sum.
    log_r_sum = max(min(log_r_sum, float(global_clip)), -float(global_clip))

    return float(log_r_sum), sum_grads

def L_CLIP_two_pass(old_model, cur_model, rm, alphas_cumprod, device, shape,
                    ddim_steps, eta, eps_to_mu_fn, clip_eps, n_episodes,
                    optimizer, grad_clip=1.0,
                    microbatch=1, per_step_clip=5.0, global_clip=10.0, sigma_floor=1e-3):
    """
    Run episodes and backpropagate the loss.
    The process is consists of two passes.
    1)

    Parameters:
        old_model (torch.nn.Module) : The fixed model updated in the previous epoch.
        cur_model (torch.nn.Module) : The model updated most recently. Parameters of cur_model keeps updated in each episodes.
        rm (torch.nn.Module) : Reward model.
        alphas_cumprod (torch.Tensor) : Tensor of alpha_bars.
        device (torch.device) : Device.
        shape (tuple) : Shape of the image.
        ddim_steps (int) : Number of steps in DDIM sampling process.
        eta (float) : The ratio of probabilistic process in the DDPM(DDIM) reverse process.
        eps_to_mu_fn (function) : Function to compute mu(mean) from epsilon.
        clip_eps (float) : Clipping value of log(r_t)*A_t in the PPO.
        n_episodes (int) : Number of episodes executed in each epoch.
        optimizer (torch.nn.optim) : Optimizer to update parameters of cur_model.
        grad_clip (float) :
        microbatch (int) : Number of episodes executed in each epoch.
        per_step_clip (float) : Clipping value of log r_t in each timestep.
        global_clip (float) : Clipping value of log_r_sum.
        simga_floor (float) : Minimum clipping value of sigma to avoid sigma to be too small.

    Returns:
        float : avg_loss_per_mb.
        dictionary : stats. Has the information about the current epoch.
    """
    # Get old_model, rm and fix their parameters.
    old_model.eval()
    rm.eval()
    for p in old_model.parameters():
        p.requires_grad = False
    for p in rm.parameters():
        p.requires_grad = False

    # Pass 1
    buffers = [] # saves steps_log(t,x_t,eps_t) and x0_cpu in each episodes.
    rewards_list = [] # saves total reward(float) in each episodes.

    # Iter episodes.
    for _ in range(n_episodes):
        with torch.no_grad():
            # Execute reverese process and get x0 and steps_log.
            x0, steps_log = sample_ddim(
                model=old_model, shape=shape,
                alphas_cumprod=alphas_cumprod, device=device,
                ddim_steps=ddim_steps, eta=eta
            )

            # Get steps_log_cpu, x0_cpu to save in the buffers list.
            x0_cpu = x0.detach().cpu()
            Tlist = steps_log["t"]
            Xlist = [xt.detach().cpu() for xt in steps_log["x_t"]]
            Elist = [et.detach().cpu() for et in steps_log["eps_t"]]
            steps_log_cpu = {"t": Tlist, "x_t": Xlist, "eps_t": Elist}

            # Get reward from the trained reward model.
            R_t = rm(x0_cpu.to(device)).mean()
            R = float(R_t.item())

        buffers.append((steps_log_cpu, x0_cpu))
        rewards_list.append(R)

    # Make it be a tensor.
    rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
    advantages = rewards.detach()

    # Prepare training.
    cur_model.train()
    total_loss_val = 0.0
    clip_cnt = 0
    sample_cnt = 0
    valid_n = len(buffers)

    # Pass 2
    for i in range(0, valid_n, microbatch):
        # Get microbatch number of episodes.
        mb = buffers[i:i+microbatch]
        adv_mb = advantages[i:i+microbatch]

        # Set optimizer.
        optimizer.zero_grad(set_to_none=True)
        params = [p for p in cur_model.parameters() if p.requires_grad]

        for (steps_log_cpu, x0_cpu), adv in zip(mb, adv_mb):
            # Compute log_r_val and sum_grads in current microbatch.
            log_r_val, sum_grads = compute_log_r_sum(
                cur_model=cur_model,
                steps_log=steps_log_cpu,
                x0_cpu=x0_cpu,
                alphas_cumprod=alphas_cumprod,
                device=device,
                eta=eta,
                eps_to_mu_fn=eps_to_mu_fn,
                per_step_clip=per_step_clip,
                global_clip=global_clip,
                sigma_floor=sigma_floor
            )

            # Get r_val and A(advantage).
            r_val = math.exp(log_r_val)
            A = float(adv.item())

            # Clip r_val as in the PPO algorithm.
            low, high = 1.0 - clip_eps, 1.0 + clip_eps
            rc = min(max(r_val, low), high)  # rc : r_clipped

            # loss = -min(r*A, rc*A)
            val  = r_val * A # val denotes the value of objective function in PPO algorithm.
            valc = rc    * A # valc : val_clipped
            chosen = val if val < valc else valc
            s = -chosen # We will use loss function, not an objective function, so multiply -1.

            # Count the number of clipped episodes in current microbatch.
            unclipped = (A >= 0 and r_val <= high) or (A < 0 and r_val >= low)
            if not unclipped:
                clip_cnt += 1
            sample_cnt += 1

            # Execute backpropagation only if the value is unclipped.
            if unclipped:
                # We want to compute (d/d_theta)(r_val)*A which is same as A*r_val*(d/d_theta)(log(r_val))
                # The value of (d/d_theta)(log(r_val)) is sum_grads, computed in the function compute_log_r_sum.
                scale = -A * r_val / float(microbatch)
                for p, g in zip(params, sum_grads):
                    grad_to_apply = (g.detach().clone().to(p.device)) * float(scale)
                    if p.grad is None:
                        p.grad = grad_to_apply
                    else:
                        p.grad.add_(grad_to_apply)

        # Gradient clip and optimizer step.
        total_norm = torch.nn.utils.clip_grad_norm_(cur_model.parameters(), grad_clip)
        optimizer.step()

    # Return the log of single epoch.
    clip_frac = (clip_cnt / sample_cnt) if sample_cnt > 0 else float("nan")
    stats = {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "episodes": int(valid_n),
        "microbatch": microbatch,
        "ddim": ddim_steps,
        "eta": eta,
        "clip": clip_eps,
        "clip_frac": float(clip_frac)
    }
    return stats

def single_epoch(ddpm_cur, ddpm_old, rm, alphas_cumprod, device, shape, optimizer,
                 eps_to_mu_fn, ddim_steps=100, eta=1.0, clip_eps=0.1,
                 episodes_per_epoch=30, normalize_rewards=True, grad_clip=1.0,
                 microbatch=1):
    """
    Run a single epoch.

    Parameters:
        ddpm_cur (torch.nn.Module) : The model updated most recently.
        ddpm_old (torch.nn.Module) : The fixed model updated in the previous epoch.
        rm (torch.nn.Module) : Reward model.
        alphas_cumprod (torch.Tensor) : Tensor of alpha_bars.
        device (torch.device) : Device.
        shape (tuple) : Shape of the image.
        optimizer (torch.nn.optim) : Optimizer to update parameters of cur_model.
        eps_to_mu_fn (function) : Function to compute mu(mean) from epsilon.
        ddim_steps (int) : Number of steps in DDIM sampling process.
        eta (float) : The ratio of probabilistic process in the DDPM(DDIM) reverse process.
        clip_eps (float) : Clipping value of log(r_t)*A_t in the PPO
        episodes_per_epoch (int) : Number of episodes executed in each epoch.
        normalize_rewards (bool) : Whether to normalize rewards.
        grad_clip (float) :
        microbatch (int) : Number of episodes executed in each epoch.

    Returns:
        dictionary : stats. Has the information about the current epoch.
    """
    ddpm_old.eval()
    for p in ddpm_old.parameters(): p.requires_grad = False
    rm.eval()
    for p in rm.parameters(): p.requires_grad = False

    stats = L_CLIP_two_pass(
        old_model=ddpm_old,
        cur_model=ddpm_cur,
        rm=rm,
        alphas_cumprod=alphas_cumprod,
        device=device,
        shape=shape,
        ddim_steps=ddim_steps,
        eta=eta,
        eps_to_mu_fn=eps_to_mu_fn,
        clip_eps=clip_eps,
        n_episodes=episodes_per_epoch,
        optimizer=optimizer,
        grad_clip=grad_clip,
        microbatch=microbatch
    )
    stats.update({"ddim_steps": ddim_steps, "eta": eta, "clip_eps": clip_eps})
    return stats