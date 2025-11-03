import yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm.auto import tqdm
from ddpmppo.reward.model import RewardModel
from ddpmppo.utils.seed import set_seed
from ddpmppo.utils.io import save_ckpt
from ddpmppo.utils.log import get_logger

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/reward.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger = get_logger()

    # 예시: 폴더 내 이미지 분류(0/1) 구조를 가정하거나, 필요시 커스터마이즈
    tfm = transforms.Compose([transforms.Resize(cfg['data']['img_size']), transforms.CenterCrop(cfg['data']['img_size']), transforms.ToTensor()])
    ds = datasets.ImageFolder(cfg['data']['root'], transform=tfm)
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    model = RewardModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 라벨은 ImageFolder의 클래스 인덱스 사용(0/1 등)
    for epoch in range(1, cfg['train']['epochs']+1):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            logit = model(x)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

    save_ckpt(model, cfg['ckpt']['out'])
    logger.info(f"saved: {cfg['ckpt']['out']}")

if __name__ == '__main__':
    main()
