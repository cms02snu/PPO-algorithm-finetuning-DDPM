import argparse
import yaml
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from reward.dataset import DeepFakeDataset
from reward.model import RewardModel
from utils import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = DeepFakeDataset(
        cfg["path"]["real_img"], 
        cfg["path"]["fake_img"], 
        num_img=cfg["train_reward"]["num_img"], 
        num_noise=cfg["train_reward"]["num_noise"],
        transform=transform)
    
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = cfg["train_reward"]["batch_size"],
        shuffle = True,
        num_workers = cfg["train_reward"]["num_workers"],
        drop_last = True
    )

    model = RewardModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train_reward"]["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    label_smoothing = cfg["train_reward"]["label_smoothing"]
    log_dir = f"logs/train_reward_model/{time.strftime('%Y%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1,cfg["train_reward"]["num_epochs"]+1):
        model.train()
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train',loss,epoch)

    writer.close()
    if cfg["train_reward"]["save"]:
        torch.save(model.state_dict(), 'rm.pt')
        print("Model saved.")

if __name__=="__main__":
    main()