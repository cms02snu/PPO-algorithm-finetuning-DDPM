# PPO-algorithm-finetuning-DDPM

This repository implements PPO algorithm to finetune trained DDPM. 

Refer to:
- Proximal Policy Optimization Algorithms (arXiv 2017)
- 3D-HLDM: Human-Guided Latent Diffusion Model to Improve Microvascular Invasion Prediction in Hepatocellular Carcinoma (IEEE 2024)

PPO algorithm is an optimizing method of reinforcement learning. We apply this algorithm in finetuning DDPM. Assume we have DDPM trained with CelebA dataset. We train reward model separately and regard DDPM as policy in the reinforcement learning as in the paper.

# PPO‑finetuning‑DDPM (PPO for Diffusion Fine‑Tuning)

A minimal, reproducible baseline for **PPO‑style fine‑tuning of a trained DDPM** with an auxiliary **reward model**.

---

## What this repo provides
- **DDPM** backbone (`ddpm/`), **PPO** loop (`ppo/`), **reward model & dataset** (`reward/`)
- **Configs** in YAML (`configs/*.yaml`) incl. a **smoke test**
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
# logs: runs/ or logs/ (TensorBoard)
```

### 3) PPO fine‑tune DDPM
```bash
python -m scripts.finetune --cfg configs/default.yaml
# key params: finetune.lr, clip_eps, episodes_per_epoch, microbatch, grad_clip
```

### 4) Evaluate
```bash
python -m scripts.eval --cfg configs/default.yaml
```

### 5) Monitor logs
```bash
tensorboard --logdir runs   # or --logdir logs
```

---

## Key config knobs (see `configs/default.yaml`)
- `finetune`: `lr`, `num_epochs`, `ddim_steps`, `eta`, `clip_eps`, `episodes_per_epoch`, `microbatch`, `grad_clip`, `save`
- `train_reward`: `num_img`, `num_noise`, `batch_size`, `num_workers`, `lr`, `label_smoothing`, `num_epochs`, `save`
- `path`: `real_img`, `fake_img`, `ddpm`, `rm`

---

## Design notes
- **Policy** = DDPM; **Reward** = learned classifier on (real,fake) images
- **Sampler**: DDIM (`ddpm/sample.py`), schedule utils in `ddpm/schedule.py`
- **Safety**: gradient clipping + PPO clipping (`ppo/algorithm.py`)
- **Determinism**: `utils.set_seed(seed, deterministic=True)`

---

## Repro recipe (example)
```bash
# quick smoke
python -m scripts.train_reward_model --cfg configs/smoke.yaml
python -m scripts.finetune --cfg configs/smoke.yaml
python -m scripts.eval --cfg configs/smoke.yaml

# full run
python -m scripts.train_reward_model --cfg configs/default.yaml
python -m scripts.finetune --cfg configs/default.yaml
python -m scripts.eval --cfg configs/default.yaml
```

---

## Results (placeholder)
Add reward curve / sample grids here.
```
runs/…/events.out.tfevents…
samples/…  # if you choose to save generated images
```

---

## FAQ
- **TensorBoard 메시지 “TensorFlow not found”**: 무시해도 됩니다. PyTorch SummaryWriter만으로 동작합니다.
- **`python -m …`를 쓰는 이유**: 패키지 상대 임포트 안정화(모듈 경로 꼬임 방지).

---

## Cite / References
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- (Domain paper you referenced)

---

## License
Add your license (e.g., MIT).

## Contact
Issues / PR 환영.

