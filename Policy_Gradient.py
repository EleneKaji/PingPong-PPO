import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

import wandb
import time
import numpy as np

# Define the Policy Gradient network
class PolicyGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(8 * 38 * 38, 128)
        self.fc2 = nn.Linear(128, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.01)

        torch.nn.init.normal_(self.conv1.bias, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc1.bias, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc2.bias, mean=0, std=0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.sigmoid(self.fc2(x))
        return x
    
def pre_process(observation):
    observation = torch.tensor(observation, dtype=torch.float32, device=device)
    
    observation = observation[:, 35:195] 
    observation = observation[:, ::2, ::2, 0] 

    mask = (observation != 144) & (observation != 109)
    observation[~mask] = 0
    observation[mask] = 1 

    return observation.unsqueeze(1)

def discount_rewards(rewards, gamma=0.99):
    num_steps, _ = rewards.size()

    discounts = torch.tensor([gamma**i for i in range(num_steps)], dtype=torch.float32, device=rewards.device)
    reversed_rewards = rewards.flip(dims=[0])
    discounted_rewards = torch.cumsum(reversed_rewards * discounts.view(-1, 1), dim=0)
    discounted_rewards = discounted_rewards.flip(dims=[0])

    discounted_rewards -= torch.mean(discounted_rewards)
    discounted_rewards /= torch.std(discounted_rewards)

    # check for nan values
    if torch.isnan(discounted_rewards).any():
        print("Nan in discounted rewards")
        discounted_rewards = torch.torch.nan_to_num(discounted_rewards, nan=0.0)
    
    return discounted_rewards

def train_one_episode():
    ep_deltas, ep_rewards = [], []
    obs, _ = envs.reset()
    done = False
    prev_obs = None

    while True:
        curr_obs = pre_process(obs)
        delta_obs = curr_obs - prev_obs if prev_obs is not None else curr_obs
        prev_obs = curr_obs

        with torch.no_grad():
            nograd_probs = model(delta_obs)
            random_value = torch.rand(nograd_probs.size(), device=device)
            action = torch.where(random_value < nograd_probs, 2, 3) # 2 is UP, 3 is DOWN

        obs, rewards, terminated, truncated, _ = envs.step(action)
        done = terminated or truncated

        ep_deltas.append(delta_obs.squeeze(0))
        ep_rewards.append(rewards)

        if done:
            d_deltas = torch.stack(ep_deltas).to(device)
            d_rewards = torch.tensor(ep_rewards, dtype=torch.float32, device=device)
            ep_deltas, ep_rewards = [], []
            
            with autocast(device_type="cuda"):
                probs = model(d_deltas)
                random_values = torch.rand(probs.size(), device=device)
                y = torch.where(random_values < probs, 1, 0) # 1 is UP, 0 is DOWN
                logprobs = y * torch.log(probs) + (1 - y) * torch.log(1 - probs)
                advantage = discount_rewards(d_rewards.unsqueeze(-1))
                loss = -torch.sum(logprobs * advantage)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1)          
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            return loss.item(), torch.sum(d_rewards).item()
        
def train_one_epoch():
    ep_deltas, ep_rewards = [], []
    obs, _ = envs.reset()
    prev_obs = None

    for _ in range(time_step):
        curr_obs = pre_process(obs) # [N, 1, 80, 80]
        delta_obs = curr_obs - prev_obs if prev_obs is not None else curr_obs
        prev_obs = curr_obs

        with torch.no_grad():
            nograd_probs = model(delta_obs)
            random_value = torch.rand(nograd_probs.size(), device=device)
            action = torch.where(random_value < nograd_probs, 2, 3) # 2 is UP, 3 is DOWN
        obs, rewards, _, _, _ = envs.step(action) 

        ep_deltas.append(delta_obs.squeeze(0)) # 1 [N, 1, 80, 80]
        ep_rewards.append(rewards)

    d_deltas = torch.cat(ep_deltas, dim=0).to(device) # [N * T, 1, 80, 80]
    ep_rewards_np = np.array(ep_rewards)
    d_rewards = torch.tensor(ep_rewards_np, dtype=torch.float32, device=device) # [T, N]
    ep_deltas.clear()
    ep_rewards.clear()
    
    with autocast(device_type="cuda"):
        probs = model(d_deltas) # [T * N]
        random_values = torch.rand(probs.size(), device=device)
        y = torch.where(random_values < probs, 1, 0) # 1 is UP, 0 is DOWN. [T * N]
        logprobs = y * torch.log(probs) + (1 - y) * torch.log(1 - probs) # torch.Size([T, N])
        advantage = discount_rewards(d_rewards).view(-1) # [T, N] -> [T * N]
        print(y.size(), advantage.size())
        loss = -torch.mean(logprobs * advantage)
        if torch.isnan(loss).any():
            print("Nan in loss")
            loss = torch.nan_to_num(loss, nan=0.0)

    scaler.scale(loss).backward()
    clip_grad_norm_(model.parameters(), max_norm=1)          
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return loss.item(), torch.mean(d_rewards).item(), torch.mean(probs).item(), torch.mean(advantage).item(), torch.mean(logprobs).item()

if __name__ == "__main__":
    plotting = True
    if plotting: wandb.init(
        project="PG-Pong",
        name="PG-2env-4096ts",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_envs = 2
    gym.register_envs(ale_py)
    envs = gym.make_vec("ALE/Pong-v5", num_envs=num_envs, vectorization_mode="async")    
    observations, info = envs.reset()

    model = PolicyGradient().to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    time_step = 4096
    total_time = 0
    for i in range(1000):
        start_time = time.time()
        loss, reward, prob, advantage, logprob = train_one_epoch()
        time_diff = time.time() - start_time
        total_time += (time_diff / 60)

        print(f"Epoch: {i} | Loss: {loss:.5g} | Reward: {reward:.5g} | Advantage: {advantage:.5g} | Prob: {prob:.5g} | LogProg: {logprob:.5g} | Time (s): {time_diff:.5g} | Total Time (m): {total_time:.5g}")
        if plotting: wandb.log({"average loss": loss, 
                                "average reward": reward, 
                                "average advantage": advantage,
                                "average prob": prob,
                                "average logprob": logprob,
                                "time": total_time}, step=i)
        if i % 100 == 0:
            torch.save(model.state_dict(), 'PGmodel.pth')

    envs.close()
    if plotting: wandb.finish()