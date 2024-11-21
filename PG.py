import gymnasium as gym
import ale_py # Arcade Learning Environment 

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

class PolicyGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=4) # 16 19 19
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2) # 32 8 8
        self.fc1 = nn.Linear(32 * 8 * 8, 256) # 256
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(-1)))
        x = torch.sigmoid(self.fc2(x))
        return x

def pre_process(observation):
    # need to check if the new pong game needs to be pre-processed different way
    observation = observation[35:195]  # crop
    observation = observation[::2, ::2, 0]  # downsample by factor of 2
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    observation = torch.tensor(observation, dtype=torch.float32) # convert to torch tensor
    observation = observation.unsqueeze(0)  # adding the channel dimension

    return observation

def plot_observation(observation):
    observation = observation.squeeze(0)  # removes the channel dimension
    
    plt.imshow(observation, cmap='gray')  # 'gray' for binary images
    plt.title("Processed Observation")
    plt.axis('off')
    plt.show()

def discount_rewards(rewards, gamma=0.99):
    discounted_r = torch.zeros_like(rewards)
    running_add = 0

    for t in reversed(range(0, rewards.size(0))):
        if rewards[t] != 0:  # reset the sum for a new episode
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    
    return discounted_r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PolicyGradient().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")
observation, info = env.reset()

dprops, drs = [], []
done = False
prev_x = None
reward_sum = 0
episode_number = 0
batch_size = 10

while not done:
    # env.render()

    # process the observation and get the difference
    curr_x = pre_process(observation).to(device)
    # plot_observation(curr_x)
    delta_x = curr_x - prev_x if prev_x is not None else torch.zeros_like(curr_x)
    prev_x = curr_x

    # get the policy probability
    policy_probs = model(delta_x)
    action = torch.multinomial(policy_probs, 1).item() # "exploration"
    action_prob = policy_probs[action].unsqueeze(0)
    dprops.append(1 - action_prob) #???

    # step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1

        epprops = torch.cat(dprops).to(device)
        epr = torch.tensor(drs).to(device)
        dprops, drs = [], []

        # get the discounted return
        discounted_epr = discount_rewards(epr)
        discounted_epr -= torch.mean(discounted_epr)
        discounted_epr /= torch.std(discounted_epr)

        loss = torch.sum(epprops * discounted_epr)
        # loss /= 10 # divide by batch size
        loss.backward()

        if episode_number % batch_size == 0:
            optimizer.step() 
            optimizer.zero_grad()

            # save the model periodically
            if episode_number % 100 == 0:
                torch.save(model.state_dict(), 'PGmodel.pth')

        # reset the environment and variables
        print(f"Episode: {episode_number}, Reward: {reward_sum}, Loss: {loss.item():.8f}")
        reward_sum = 0
        observation, info = env.reset()
        prev_x = None
        done = False

    if episode_number == 2000:
        break

# env.close()

"""
small pseudocode

get the observation fro env.reset()
while true:
    pre-process the observation
    get the change from prev to curr observation
    
    get the current policy probability (policy network forward)
        this is done by comparing the probability to random number, not a log probability?
    determine the new action based on the policy probability, which can be done either up or down
        represented as y so its either 0 and 1, if the action is any number oustide of it
    calculate the probability for that step and append to all probabilities
        by subtracting y and prob
    
    step the environment and get observation, reward, done, info
    add reward to reward sum
    append reward

    if done: truncated ot terminated because of env issue or because it is the end of the episode
        add to episode sum
        
        get the discounted return 

"""