import argparse, yaml, torch
from pathlib import Path
from torchvision.utils import save_image
from ddpmppo.utils.io import load_state_dict_maybe_wrapped
from ddpmppo.diffusion.ddpm import DDPMModel
from ddpmppo.diffusion.sampling import sample_ddim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/sample.yaml')
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    device = torch.device(cfg.get('device','cuda') if torch.cuda.is_available() else 'cpu')
    model = DDPMModel().to(device)
    ckpt_path = args.ckpt or cfg['ckpt']['ddpm']
    load_state_dict_maybe_wrapped(model, ckpt_path)

    imgs = sample_ddim(model, num_images=cfg['out']['num_images'], img_size=cfg['out']['img_size'], device=device)
    out_dir = Path(cfg['out']['dir']); out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(imgs.size(0)):
        save_image(imgs[i], out_dir / f"sample_{i:03d}.png")
    print(f"saved to: {out_dir}")

if __name__ == '__main__':
    main()
