import gymnasium as gym
import DTRGym  # This will test if the package is registered

env = gym.make("AhnChemoEnv-discrete", n_act=5)

print("Environment loaded successfully!")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
