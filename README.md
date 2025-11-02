# PPO-algorithm-finetuning-DDPM

This repository implements PPO algorithm to finetune trained DDPM. 

Refer to:
- Proximal Policy Optimization Algorithms (arXiv 2017)
- 3D-HLDM: Human-Guided Latent Diffusion Model to Improve Microvascular Invasion Prediction in Hepatocellular Carcinoma (IEEE 2024)

PPO algorithm is a optimizing method of reinforcement learning. We apply this algorithm in finetuning DDPM. Assume we have DDPM trained with CelebA dataset. We train reward model separately and regard DDPM as policy in the reinforcement learning as in the paper.
