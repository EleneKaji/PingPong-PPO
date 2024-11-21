import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")

num_actions = env.action_space.n
print(f"Number of actions available: {num_actions}")

# Get the action meanings (if supported)
if hasattr(env.unwrapped, 'get_action_meanings'):
    action_meanings = env.unwrapped.get_action_meanings()
    print("Action meanings:")
    for idx, action in enumerate(action_meanings):
        print(f"Action {idx}: {action}")
else:
    print("Action meanings not available for this environment.")