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

def discount_rewards(rewards, gamma=0.99):
    # Assume rewards is a 2D tensor of shape [num_steps, num_envs]
    num_steps, num_envs = rewards.size()
    
    # Create a discount factor tensor with shape [num_steps]
    discounts = torch.tensor([gamma**i for i in range(num_steps)], dtype=torch.float32, device=rewards.device)
    print(discounts)
    
    # Reverse the rewards to use cumulative sum effectively
    reversed_rewards = rewards.flip(dims=[0])
    print(reversed_rewards)
    
    # Apply cumulative sum along the time dimension (0)
    discounted_rewards = torch.cumsum(reversed_rewards * discounts.view(-1, 1), dim=0)
    print(discounted_rewards)
    
    # Flip back to the original order
    discounted_rewards = discounted_rewards.flip(dims=[0])
    print(discounted_rewards)
    
    # Normalize by dividing by the discount factors
    print(torch.mean(discounted_rewards))
    discounted_rewards -= torch.mean(discounted_rewards)
    discounted_rewards /= torch.std(discounted_rewards)
    print(discounted_rewards)
    
    return discounted_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
    [26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35],
    [36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45],
    [46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55],
    [56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65],
    [66, 67, 68, 69, 70],
    [71, 72, 73, 74, 75],
    [76, 77, 78, 79, 80],
    [81, 82, 83, 84, 85],
    [86, 87, 88, 89, 90],
    [91, 92, 93, 94, 95],
    [96, 97, 98, 99, 100]
])

discount_rewards(tensor, gamma=0.99)