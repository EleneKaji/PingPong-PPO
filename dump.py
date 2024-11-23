# import gymnasium as gym
# import ale_py

# gym.register_envs(ale_py)
# env = gym.make("ALE/Pong-v5", render_mode="human")

# num_actions = env.action_space.n
# print(f"Number of actions available: {num_actions}")

# # Get the action meanings (if supported)
# if hasattr(env.unwrapped, 'get_action_meanings'):
#     action_meanings = env.unwrapped.get_action_meanings()
#     print("Action meanings:")
#     for idx, action in enumerate(action_meanings):
#         print(f"Action {idx}: {action}")
# else:
#     print("Action meanings not available for this environment.")

import torch

# def discount_rewards(rewards, gamma=0.99):
#     # Assume rewards is a 2D tensor of shape [num_steps, num_envs]
#     num_steps, num_envs = rewards.size()
    
#     # Create a discount factor tensor with shape [num_steps]
#     discounts = torch.tensor([gamma**i for i in range(num_steps)], dtype=torch.float32, device=rewards.device)
#     print(discounts)
    
#     # Reverse the rewards to use cumulative sum effectively
#     reversed_rewards = rewards.flip(dims=[0])
#     print(reversed_rewards)
    
#     # Apply cumulative sum along the time dimension (0)
#     discounted_rewards = torch.cumsum(reversed_rewards * discounts.view(-1, 1), dim=0)
#     print(discounted_rewards)
    
#     # Flip back to the original order
#     discounted_rewards = discounted_rewards.flip(dims=[0])
#     print(discounted_rewards)
    
#     # Normalize by dividing by the discount factors
#     print(torch.mean(discounted_rewards))
#     discounted_rewards -= torch.mean(discounted_rewards)
#     discounted_rewards /= torch.std(discounted_rewards)
#     print(discounted_rewards)
    
#     return discounted_rewards

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor = torch.tensor([
#     [1, 2, 3, 4, 5],
#     [6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15],
#     [16, 17, 18, 19, 20],
#     [21, 22, 23, 24, 25],
#     [26, 27, 28, 29, 30],
#     [31, 32, 33, 34, 35],
#     [36, 37, 38, 39, 40],
#     [41, 42, 43, 44, 45],
#     [46, 47, 48, 49, 50],
#     [51, 52, 53, 54, 55],
#     [56, 57, 58, 59, 60],
#     [61, 62, 63, 64, 65],
#     [66, 67, 68, 69, 70],
#     [71, 72, 73, 74, 75],
#     [76, 77, 78, 79, 80],
#     [81, 82, 83, 84, 85],
#     [86, 87, 88, 89, 90],
#     [91, 92, 93, 94, 95],
#     [96, 97, 98, 99, 100]
# ])

# discount_rewards(tensor, gamma=0.99)

import matplotlib.pyplot as plt

# List of losses provided
losses = [
    -6.41398068, 3.84917767, -20.80338456, 3.60161982, 5.01723531, -6.73300803,
    -4.07163446, -2.96727184, -6.67661046, 0.94222783, 11.04009719, 12.18560668,
    -3.95145959, 3.20739586, 11.27573348, 9.74817368, -9.85942466, 7.67840353,
    9.94652233, 9.30912470, 5.89822059, -7.48104157, -9.82382639, -4.11239441,
    -2.22976973, 20.50706190, -9.83215618, -14.64978417, 15.36855079, -22.07929273,
    15.03871815, 11.90582887, -3.13344116, -6.77200663, -6.29397970, -13.74697922,
    9.58430697, -0.81539202, -7.21410014, -12.36066670, 6.15275403, -0.82628393,
    1.57217892, -2.47360486, 14.35365696, 1.45342304, -0.71221341, -11.25543233,
    -32.36120101, -5.05316357, 3.25217120, -7.78976531, 14.23557610, -3.36859403,
    8.63452786, -5.35905357,
-4.21724485,
-12.03520252,
-7.03856447,
-13.72887537,
13.91418101,
1.86938667,
11.56428584,
-1.64506151,
-2.22540128,
-7.31949364,
-18.37153291,
-1.76174871,
-8.86735697,
-12.78784318,
-9.79013578,
-2.16420682,
-16.31236651,
3.11358525,
-9.89100354,
-10.48160110,
2.35702550,
-10.60293150,
-0.37566048,
-21.26435906,
7.95601570,
-8.65782510,
-4.73969005,
-2.83338028,
-0.10841919,
14.32235144,
-1.69581804,
20.31798173,
-26.27463791,
0.02051941,
22.42142509,
-1.19610007,
-11.50631594
]

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Loss')
plt.title('Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
