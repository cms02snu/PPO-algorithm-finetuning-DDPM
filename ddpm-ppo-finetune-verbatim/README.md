# ddpm-ppo-finetune

**목적**: 이미 학습된 DDPM 파라미터(체크포인트)를 불러와 **PPO**로 보상 기반 파인튜닝하는 표준형 레포입니다.  
(기본 DDPM *훈련 코드는 포함하지 않음* — 제공된 체크포인트만 사용)

## 설치

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## 실행 개요

- 보상모델 학습(선택): `train-reward --cfg configs/reward.yaml`
- DDPM 파인튜닝: `finetune-ddpm --cfg configs/finetune.yaml`
- 샘플 생성: `sample-ddpm --cfg configs/sample.yaml --ckpt checkpoints/finetuned/epoch_001.pth`

> 체크포인트/데이터 경로는 `configs/*.yaml`에서 바꾸세요.

## 폴더 구조

```
ddpm-ppo-finetune/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ configs/
├─ src/ddpmppo/
│  ├─ cli/
│  ├─ diffusion/
│  ├─ reward/
│  ├─ ppo/
│  ├─ utils/
│  └─ legacy/        # 업로드 하신 원본 Colab 코드 보관
├─ checkpoints/      # (gitignore)
└─ data/             # (gitignore)
```

## 주의
- 이 레포는 **이미 학습된 DDPM**을 입력으로 가정합니다(`configs/finetune.yaml`의 `ckpt.base_ddpm`).
- Colab 전용 코드(`drive.mount`, 경로 등)는 `legacy/`에 보관만 하고, 실행 파이프라인은 `src/ddpmppo/*`에 새로 구성했습니다.
