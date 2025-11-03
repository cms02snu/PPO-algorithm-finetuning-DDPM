import torch

@torch.no_grad()
def sample_ddim(model, num_images=8, img_size=64, device='cuda'):
    # (데모용) 가우시안에서 한 번에 생성 — 실제 DDIM이 아님.
    x = torch.randn(num_images, 3, img_size, img_size, device=device)
    # 통상은 DDIM 스텝 루프에서 model을 여러 번 호출하여 x를 업데이트함.
    return (x.clamp(-1,1) + 1) * 0.5  # [0,1] 범위로 근사
