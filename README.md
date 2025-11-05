# PPO-finetuning-DDPM

A minimal, reproducible baseline for **PPO-style fine-tuning of a trained DDPM** with an auxiliary **reward model**.

---

## What this repo provides
- **DDPM** backbone (`ddpm/`), **PPO** loop (`ppo/`), **reward model & dataset** (`reward/`)
- **Configs** in YAML (`configs/*.yaml`) including a **smoke test**
- **Scripts**: train reward model → PPO fine-tune → eval
- **TensorBoard** logging (via `SummaryWriter`)
- Reproducibility: `set_seed()`

```
repo/
├─ configs/            # default.yaml, smoke.yaml
├─ ddpm/               # model, schedule, sampling
├─ ppo/                # algorithm, utils
├─ reward/             # dataset, reward model
├─ scripts/            # train_reward_model.py, finetune.py, eval.py
├─ utils.py
└─ requirements.txt
```

---

## Quickstart
### 0) Env
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1) Data & weights
```
data/real/*   data/fake/*     # images for reward model
ddpm.pth      # (optional) pretrained DDPM weights
```
Set paths in `configs/default.yaml` (or use `configs/smoke.yaml` to sanity-check CPU execution).

### 2) Train reward model
```bash
python -m scripts.train_reward_model --cfg configs/default.yaml
```

### 3) PPO fine-tune DDPM
```bash
python -m scripts.finetune --cfg configs/default.yaml
```

### 4) Evaluate
```bash
python -m scripts.eval --cfg configs/default.yaml
```

### 5) Monitor logs
```bash
tensorboard --logdir logs   # or --logdir runs
```

---

## PPO $$L_{\text{CLIP}}$$ configuration (this implementation)

> We treat the DDPM as the **policy** and the entire DDIM reverse process as **one episode**. The reward is computed by a **frozen Reward Model** on the final $$x_0$$.

### Objective
$$
L^{\text{CLIP}}(\theta)
= \hat{\mathbb{E}}\left[
\min\big(r(\theta)\,\hat A,\ \mathrm{clip}(r(\theta),\,1-\varepsilon,\,1+\varepsilon)\,\hat A\big)
\right]
$$

with

- $$r(\theta)=\tfrac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$$, and a practical estimator via per-step log-ratio: $ \log r(\theta)=\sum_{t=1}^T \log r_t,\qquad
\log r_t \propto
\frac{\|x_{t-1}-\mu_{\mathrm{old}}\|^2-\|x_{t-1}-\mu_{\theta}\|^2}{2\,\tilde{\beta}_t} $

- $$\hat A$$: advantage from the Reward Model on $$x_0$$ (non-trainable in this repo).

### Implementation notes (two-pass, memory-friendly)
1. **Pass-1 (no-grad):** run the entire DDIM reverse process; cache minimal per-step stats and the final $$\hat A$$.  
2. **Pass-2 (with-grad):** accumulate $$\sum_t \nabla_\theta \log r_t$$ step-by-step, applying `per_step_clip` and `global_clip`.  
3. **Clipped branch handling:** use the *min* form for the loss value, but **apply gradients only when unclipped** (to reduce boundary bias).  
4. **Stability:** apply `sigma_floor` and `grad_clip`; average gradients over `microbatch` before `optimizer.step()`.
