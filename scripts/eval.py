import argparse
import yaml
import torch
from utils import set_seed
from reward.model import RewardModel
from ddpm.model import DDPMModel
from ddpm.sample import sample_ddim
from ddpm.schedule import get_alpha_bar

def test(ddpm,rm,device,ddim_steps,eta):
    with torch.no_grad():
        x0,_ = sample_ddim(
            model=ddpm,
            shape=(64,3,64,64),
            alphas_cumprod=get_alpha_bar(),
            device=device,
            ddim_steps=ddim_steps,
            eta=eta
        )
        reward = rm(x0).mean().item()
        
    return reward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    rm = RewardModel().to(device)
    if cfg["path"]["rm"]:
        rm_ckpt_path = cfg["path"]["rm"]
        rm_state = torch.load(rm_ckpt_path, map_location="cpu")
        rm.load_state_dict(rm_state["model_state_dict"] if "model_state_dict" in rm_state else rm_state)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False

    ddpm = DDPMModel().to(device)
    if cfg["path"]["ddpm"]:
        ddpm_ckpt_path = cfg["path"]["ddpm"]
        ddpm_state = torch.load(ddpm_ckpt_path, map_location="cpu")
        ddpm.load_state_dict(ddpm_state["model_state_dict"] if "model_state_dict" in ddpm_state else ddpm_state)
    ddpm.eval()
    for p in ddpm.parameters():
        p.requires_grad = False

    ddim_steps = cfg["finetune"]["ddim_steps"]
    eta = cfg["finetune"]["eta"]

    reward = test(ddpm,rm,device,ddim_steps,eta)
    print(reward)

if __name__=="__main__":
    main()