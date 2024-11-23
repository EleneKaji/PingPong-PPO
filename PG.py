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
    # discounted_rewards /= torch.std(discounted_rewards)
    
    return discounted_rewards

"""------------------------------------------------------------------------------------------------------------------"""

if __name__ == "__main__":

    wandb.init(
        project="PG-Pong",
        name="PG-16envs",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PolicyGradient().to(device)
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_envs = 16
    gym.register_envs(ale_py)
    envs = gym.make_vec("ALE/Pong-v5", num_envs=num_envs, vectorization_mode="sync")
    observations, info = envs.reset()

    dprops, drs = [], []
    done = [False] * num_envs
    prev_x = [None] * num_envs
    reward_sum = [0] * num_envs
    episode_number = 0

    loss = 0

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
            random_values = torch.rand(policy_probs.size(0), device=device)
            actions = torch.where(random_values.unsqueeze(1) < policy_probs, torch.tensor(2).to(device), torch.tensor(3).to(device))
        dprops.append(policy_probs) 

        observations, rewards, terminateds, truncateds, _ = envs.step(actions)
        done = any(terminateds) or any(truncateds) 

        reward_sum += rewards
        drs.append(rewards)

        if done:
            episode_number += num_envs

            epprops = torch.stack(dprops).squeeze(-1)
            epr = torch.tensor(np.concatenate(drs)).view(-1, num_envs).to(device)

            dprops.clear() 
            drs.clear() 
            
            discounted_epr = discount_rewards(epr)
            
            with autocast(device_type="cuda"):
                loss = torch.mean(torch.log(epprops) * discounted_epr)
            scaler.scale(loss).backward()
            
            # print(clip_grad_norm_(model.parameters(), max_norm=1))            
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            # save the model periodically
            if episode_number % (100 * num_envs) == 0:
                torch.save(model.state_dict(), 'PGmodel.pth')

            end_time = time.time()
            time_diff_ms = (end_time - start_time) * 1000
            start_time = time.time()

            # reset the environment and variables
            average_reward = reward_sum.mean()
            print(f"Episode: {episode_number}, Reward: {average_reward}, Loss: {loss.item():.8f}, Time: {time_diff_ms:.2f}ms")
            wandb.log({"loss": loss, "average reward": average_reward}, step=episode_number)
            reward_sum = [0] * num_envs
            observations, info = envs.reset()
            prev_x = [None] * num_envs
            done = [False] * num_envs

    envs.close()
    wandb.finish()