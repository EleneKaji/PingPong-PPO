import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
from gymnasium.wrappers import RecordVideo

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

# Preprocess observations
def preprocess_obs(obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = obs.squeeze(0)
    obs = obs[35:195] 
    obs = obs[::2, ::2, 0]  
    obs[obs == 144] = 0
    obs[obs == 109] = 0
    obs[obs != 0] = 1
    return obs

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: True)

agent = PGAgent()
agent.load_state_dict(torch.load('PGmodel.pth', weights_only=True))
agent.eval()

observation, info = env.reset()
prev_obs = None
reward_sum = 0

try: 
    while True:
        curr_obs = preprocess_obs(observation)
        delta_x = curr_obs - prev_obs if prev_obs is not None else torch.zeros_like(curr_obs)
        delta_x = delta_x.reshape(1, -1)
        prev_obs = curr_obs
        
        with torch.no_grad():
            probs = agent(delta_x)
            action = torch.multinomial(probs, num_samples=1).item()

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        reward_sum += reward

        if done:
            print(f"Episode reward: {reward_sum}")
            observation, info = env.reset()
            prev_obs = None
            reward_sum = 0

except KeyboardInterrupt:
    print("Done!")

finally:
    env.close()
