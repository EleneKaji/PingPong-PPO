import gymnasium as gym
import ale_py # Arcade Learning Environment 

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

class PolicyGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2) # 16 38 38
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2) # 32 17 17
        self.fc1 = nn.Linear(32 * 17 * 17, 256) # 256
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1))) # flatten based on the batches
        x = F.sigmoid(self.fc2(x))
        return x

def pre_process(observations):
    processed_observations = []

    for observation in observations:
        # Pre-process each observation
        observation = observation[35:195]  # crop
        observation = observation[::2, ::2, 0]  # downsample by factor of 2
        observation[observation == 144] = 0  # erase background (background type 1)
        observation[observation == 109] = 0  # erase background (background type 2)
        observation[observation != 0] = 1  # everything else (paddles, ball) set to 1
        
        # Convert to PyTorch tensor and add channel dimension
        observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.unsqueeze(0)  # Add channel dimension

        processed_observations.append(observation)

    # Concatenate all processed observations along the batch dimension
    return torch.stack(processed_observations)

def plot_observation(observation):
    observation = observation.squeeze(0).cpu()  # removes the channel dimension
    ax.imshow(observation, cmap='gray')  # 'gray' for binary images
    ax.set_title("Processed Observation")
    ax.axis('off')
    plt.draw()
    plt.pause(0.001)

def discount_rewards(rewards, gamma=0.99):
    num_steps, num_envs = rewards.size()

    discounts = torch.tensor([gamma**i for i in range(num_steps)], dtype=torch.float32, device=rewards.device)
    reversed_rewards = rewards.flip(dims=[0])
    discounted_rewards = torch.cumsum(reversed_rewards * discounts.view(-1, 1), dim=0)
    discounted_rewards = discounted_rewards.flip(dims=[0])
    
    # Normalize by dividing by the discount factors
    # discounted_rewards /= discounts.view(-1, 1)

    discounted_rewards -= torch.mean(discounted_rewards)
    discounted_rewards /= torch.std(discounted_rewards)
    
    return discounted_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PolicyGradient().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_envs = 2
gym.register_envs(ale_py)
envs = gym.make_vec("ALE/Pong-v5", num_envs=num_envs, vectorization_mode="sync")
observations, info = envs.reset()

dprops, drs = [], []
done = [False] * num_envs
prev_x = [None] * num_envs
reward_sum = [0] * num_envs
episode_number = 0
batch_size = 2

# plt.ion()
# fig, ax = plt.subplots()

while episode_number < 20:
    # env.render()

    # print("Processing observations...")
    # process the observation and get the difference
    curr_x = pre_process(observations)
    if prev_x[0] is None:
        prev_x = torch.zeros_like(curr_x)
    
    delta_x = curr_x - prev_x
    prev_x = curr_x

    # print("Getting policy probabilities...")
    # get the policy probability
    policy_probs = model(delta_x.to(device))
    policy_probs = policy_probs.to("cpu")
    # print("policy_probs", policy_probs)
    random_values = torch.rand(policy_probs.size(0))  # Same size as batch_size

    # print("Selecting actions...")
    # For each environment in the batch, select the action based on the random value
    actions = torch.where(random_values.unsqueeze(1) < policy_probs, torch.tensor(2), torch.tensor(3))    
    ys = torch.where(actions == 2, torch.tensor(1), torch.tensor(0))  # 1 for UP, 0 for DOWN
    # print("prob", ys - policy_probs)
    dprops.append((ys - policy_probs).squeeze(1))

    # print("Stepping the environment...")
    # step the environment
    observations, rewards, terminateds, truncateds, _ = envs.step(actions)
    done = all(terminateds) or all(truncateds) # is this have to be done for any of the environments?

    reward_sum += rewards
    drs.append(rewards)
    # print(dprops[0].device, drs[0].device)

    # print("dprops", len(dprops), dprops[0])
    # print("drs", len(drs), drs[0])

    if done:
        print("done")
        episode_number += 1

        epprops = torch.stack(dprops).to(device)
        epr = torch.tensor(np.concatenate(drs)).view(-1, num_envs).to(device)

        dprops.clear() 
        drs.clear() 

        discounted_epr = discount_rewards(epr)
        loss = torch.sum(epprops * discounted_epr)
        loss /= batch_size # divide by batch size
        loss.backward()

        if episode_number % batch_size == 0:
            optimizer.step() 
            optimizer.zero_grad()

            # save the model periodically
            if episode_number % 100 == 0:
                torch.save(model.state_dict(), 'PGmodel.pth')

        # reset the environment and variables
        print(f"Episode: {episode_number}, Reward: {reward_sum/num_envs}, Loss: {loss.item():.8f}")
        reward_sum = [0] * num_envs
        observations, info = envs.reset()
        prev_x = [None] * num_envs
        done = [False] * num_envs

    torch.cuda.empty_cache()

# plt.ioff()
# plt.show()
envs.close()