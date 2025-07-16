import gymnasium as gym
import DTRGym

env = gym.make("AhnChemoEnv-discrete", n_act=5)
obs = env.reset()[0]

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # random drug
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    print(f"Step: Action={action}, Reward={reward}, Obs={obs}")

print("Episode finished. Total Reward:", total_reward)

