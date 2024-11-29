import torch
import torch.nn as nn

import gymnasium as gym
import ale_py

import wandb
import time

class PGAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(80 * 80, 256)
        self.ln2 = nn.Linear(256, 128)
        self.ln3 = nn.Linear(128, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.ln1(x))
        x = self.relu(self.ln2(x))
        x = self.ln3(x)
        return self.softmax(x)
    
def preprocess_obs(obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = obs.squeeze(0)
    obs = obs[35:195] 
    obs = obs[::2,::2,0]
    obs[obs == 144] = 0 
    obs[obs == 109] = 0 
    obs[obs != 0] = 1 
    return obs

def discount_rewards(rewards, gamma=0.99):
    discounted_r = torch.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] != 0: 
            running_add = 0 
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

gym.register_envs(ale_py)
envs = gym.make("ALE/Pong-v5")
observation, info = envs.reset()

plotting = True
if plotting: wandb.init(
    project="PG-Pong",
    name="PG-1env-4batch-simple",
)

agent = PGAgent()
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

prev_obs = None
all_obs = []
all_rewards = []
all_actions = []
batch_size = 4
episode_num = 0

start_time = time.time()
total_time = 0
while True:
    curr_obs = preprocess_obs(observation)
    delta_x = curr_obs - prev_obs if prev_obs is not None else torch.zeros_like(curr_obs)
    delta_x = delta_x.reshape(1, -1)
    prev_obs = curr_obs

    with torch.no_grad():
        probs = agent(delta_x)
        action = torch.multinomial(probs, num_samples=1).item()

    observation, reward, terminated, truncated, info = envs.step(action)
    done = terminated | truncated

    all_obs.append(delta_x)
    all_rewards.append(reward)
    all_actions.append(action)

    if done:
        episode_num += 1
    
        if episode_num % batch_size == 0 and episode_num > 0:
            obss = torch.stack(all_obs).squeeze(1)
            rewards = torch.tensor(all_rewards, dtype=torch.float32)

            logits = agent(obss)
            log_probs = torch.log(logits)
            selected_actions = torch.tensor(all_actions, dtype=torch.long) 
            action_log_probs = log_probs[torch.arange(len(selected_actions)), selected_actions]
            
            discounted_rewards = discount_rewards(rewards)
            discounted_rewards -= discounted_rewards.mean()
            discounted_rewards /= (discounted_rewards.std() + 1e-8)

            loss = -torch.mean(action_log_probs * discounted_rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_obs = []
            all_rewards = []
            all_actions = []
            prev_obs = None

            max_prob = torch.max(logits).item()
            max_log_prob = torch.max(log_probs).item()
            mean_advantage = torch.mean(discounted_rewards).item()
            time_diff = time.time() - start_time
            total_time += (time_diff / 60)
            start_time = time.time()
            average_reward = rewards.sum().item() / batch_size

            grad_norm = 0.0
            for param in agent.parameters():
                if param.grad is not None: 
                    param_norm = param.grad.data.norm(2) 
                    grad_norm += param_norm.item() ** 2

            grad_norm = grad_norm ** 0.5

            print(f"Episode: {episode_num}, Loss: {loss.item()}, Reward: {average_reward}, Advantage: {mean_advantage}, Max Prob: {max_prob}, Max Log Prob: {max_log_prob}")

            if plotting: wandb.log({"loss": loss, 
                                    "average reward": average_reward, 
                                    "average advantage": mean_advantage,
                                    "max prob": max_prob,
                                    "max logprob": max_log_prob,
                                    "grad norm": grad_norm,
                                    "time": total_time}, step=episode_num)
            if episode_num % 100 == 0: torch.save(agent.state_dict(), 'PGmodel.pth')

        done = False
        observation, info = envs.reset()

env.close()
if plotting: wandb.finish()