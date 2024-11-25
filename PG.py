import gymnasium as gym
import ale_py # Arcade Learning Environment 

import numpy as np
import time
import wandb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt

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
        x = F.relu(self.fc1(x.view(x.size(0), -1))) # flatten based on the batches
        x = F.sigmoid(self.fc2(x))
        return x

def pre_process(observations):
    observations = torch.tensor(observations, dtype=torch.float32, device=device)
    
    observations = observations[:, 35:195] 
    observations = observations[:, ::2, ::2, 0] 

    mask = (observations != 144) & (observations != 109)
    observations[~mask] = 0
    observations[mask] = 1 

    return observations.unsqueeze(1)

def discount_rewards(rewards, gamma=0.99):
    num_steps, num_envs = rewards.size()

    discounts = torch.tensor([gamma**i for i in range(num_steps)], dtype=torch.float32, device=rewards.device)
    reversed_rewards = rewards.flip(dims=[0])
    discounted_rewards = torch.cumsum(reversed_rewards * discounts.view(-1, 1), dim=0)
    discounted_rewards = discounted_rewards.flip(dims=[0])

    discounted_rewards -= torch.mean(discounted_rewards)
    discounted_rewards /= torch.std(discounted_rewards)

    # check for nan values
    if torch.isnan(discounted_rewards).any():
        print("Nan in discounted rewards")
        discount_rewards = torch.torch.nan_to_num(discount_rewards, nan=0.0)
    
    return discounted_rewards

"""------------------------------------------------------------------------------------------------------------------"""

if __name__ == "__main__":

    plotting = True

    if plotting: wandb.init(
        project="PG-Pong",
        name="PG-final-async",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PolicyGradient().to(device)
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_envs = 16
    gym.register_envs(ale_py)
    envs = gym.make_vec("ALE/Pong-v5", num_envs=num_envs, vectorization_mode="async")
    observations, info = envs.reset()

    dprops, drs, dys, drprops = [], [], [], []
    dones = [False] * num_envs
    prev_x = [None] * num_envs
    reward_sum = [0] * num_envs
    episode_number = 0

    loss = 0
    total_time = 0

    start_time = time.time()

    while episode_number < (5000 * num_envs):

        # process the observation and get the difference
        curr_x = pre_process(observations)
        if prev_x[0] is None:
            prev_x = torch.zeros_like(curr_x)
        
        delta_x = curr_x - prev_x
        prev_x = curr_x

        with autocast(device_type="cuda"):
            policy_probs = model(delta_x)

        drprops.append(policy_probs)
        random_values = torch.rand(policy_probs.size(0), device=device)
        actions = torch.where(random_values.unsqueeze(1) < policy_probs, 2, 3)
        y = torch.where(actions == 2, 1, 0) # 1 is up, 0 is down
        log_prob = y * torch.log(policy_probs) + (1 - y) * torch.log((1 - policy_probs))

        # check for nan values
        if torch.isnan(log_prob).any():
            print("Nan in log_prob")
            log_prob = torch.torch.nan_to_num(log_prob, nan=0.0)
        
        dprops.append(log_prob) 

        envs.step_async(actions)
        observations, rewards, terminated, truncateds, _ = envs.step_wait()
        dones = terminated | truncateds

        reward_sum += rewards
        drs.append(rewards)

        if np.any(dones):
            done_envs = np.sum(dones)
            episode_number += done_envs

            epprops = torch.stack(dprops).squeeze(-1).to(device)
            erprops = torch.stack(drprops).squeeze(-1).to(device)
            epr = torch.tensor(np.concatenate(drs)).view(-1, num_envs).to(device)

            dprops.clear() 
            drs.clear() 
            
            discounted_epr = discount_rewards(epr)
            
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                loss = -torch.mean(epprops * discounted_epr)

            # check for nan values
            if torch.isnan(loss).any():
                print("Nan in loss")
                loss = torch.nan_to_num(loss, nan=0.0)

            scaler.scale(loss).backward()
            
            clip_grad_norm_(model.parameters(), max_norm=1)          
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            # save the model periodically
            if episode_number % 500 == 0:
                torch.save(model.state_dict(), 'PGmodel.pth')

            end_time = time.time()
            time_diff = end_time - start_time
            total_time += (time_diff / 60)
            start_time = time.time()

            average_adv = discounted_epr.mean()
            average_logs = epprops.mean()
            average_logits = erprops.mean()

            average_reward = reward_sum.mean()
            print(f"Episode: {episode_number}, Reward: {average_reward}, Advantage: {average_adv}, Logits: {average_logits}, BCE: {average_logs}, Loss: {loss.item():.8f}, Time: {time_diff:.2f}s, Total Time: {total_time:.2f}m")

            if plotting: wandb.log({"loss": loss, 
                    "average reward": average_reward, 
                    "average advantage": average_adv, 
                    "average probabilities": average_logits,
                    "average log probabilities": average_logs,
                    "time": total_time}, step=episode_number)
            
            reward_sum = [0] * num_envs
            done_indices = torch.tensor(np.where(dones)[0], device=device)
            prev_x[done_indices] = torch.zeros_like(prev_x[done_indices], device=device)

            observations, rewards, terminated, truncateds, _ = envs.step(actions)
            dones = terminated | truncateds
            
            
    torch.save(model.state_dict(), 'PG_FinalModel.pth')
    envs.close()
    if plotting: wandb.finish()