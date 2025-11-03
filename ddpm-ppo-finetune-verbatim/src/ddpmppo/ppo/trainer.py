import torch
from tqdm.auto import tqdm
from ddpmppo.ppo.objective import ppo_clip_objective

def finetune_one_epoch(
    ddpm_cur, ddpm_old, reward_model, device,
    episodes_per_epoch=1, microbatch=1, clip_eps=0.2,
    grad_clip=1.0, **kwargs
):
    ddpm_cur.train()
    ddpm_old.eval(); 
    for p in ddpm_old.parameters(): p.requires_grad = False
    reward_model.eval()
    for p in reward_model.parameters(): p.requires_grad = False

    opt = kwargs.get('optimizer', torch.optim.Adam(ddpm_cur.parameters(), lr=1e-6))
    stats = {}

    # (데모) 무작위 노이즈 → 한 번의 forward → 보상 → 간단한 PPO step
    # 실제 구현에서는 DDIM 루프, per-timestep log prob 등 세밀한 로직이 필요
    for ep in tqdm(range(episodes_per_epoch), desc="episodes"):
        # pass1: 샘플 생성(grad off)
        with torch.no_grad():
            x = torch.randn(4, 3, 64, 64, device=device)  # fake noisy
            eps_new = ddpm_cur(x, None)
            eps_old = ddpm_old(x, None)
            # 로그확률 근사치(데모): 음의 L2를 logprob 근사로 사용
            logprob_new = -((eps_new - x)**2).mean(dim=(1,2,3))
            logprob_old = -((eps_old - x)**2).mean(dim=(1,2,3))
            ratio = torch.exp(logprob_new - logprob_old)

            # 보상(데모): reward_model 점수의 sigmoid로 유틸리티 근사
            rew = torch.sigmoid(reward_model(x)).detach()
            adv = rew - rew.mean()  # 아주 단순한 advantage 근사

        # pass2: PPO 역전파
        opt.zero_grad()
        eps_new2 = ddpm_cur(x, None)
        logprob_new2 = -((eps_new2 - x)**2).mean(dim=(1,2,3))
        ratio2 = torch.exp(logprob_new2 - logprob_old.detach())
        loss = ppo_clip_objective(ratio2, adv, eps=clip_eps)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(ddpm_cur.parameters(), grad_clip)
        opt.step()

        stats = {"loss": float(loss.item()), "ratio_mean": float(ratio2.mean().item())}
    return stats
