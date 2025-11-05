# PPO‑finetuning‑DDPM

A minimal, reproducible baseline for **PPO‑style fine‑tuning of a trained DDPM** with an auxiliary **reward model**.

---

## What this repo provides
- **DDPM** backbone (`ddpm/`), **PPO** loop (`ppo/`), **reward model & dataset** (`reward/`)
- **Configs** in YAML (`configs/*.yaml`) including a **smoke test**
- **Scripts**: train reward model → PPO fine‑tune → eval
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
Set paths in `configs/default.yaml` (or use `configs/smoke.yaml` to sanity‑check CPU execution).

### 2) Train reward model
```bash
python -m scripts.train_reward_model --cfg configs/default.yaml
```

### 3) PPO fine‑tune DDPM
```bash
python -m scripts.finetune --cfg configs/default.yaml
```

### 4) Evaluate
```bash
python -m scripts.eval --cfg configs/default.yaml
```

### 5) Monitor logs
```bash
tensorboard --logdir logs
```

---
