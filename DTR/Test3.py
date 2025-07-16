import gymnasium as gym
import DTRGym  # Required to register the environments
import numpy as np

# Create the chemotherapy environment (discrete action space with 5 actions)
env = gym.make("AhnChemoEnv-discrete", n_act=5)

# Reset the environment to get the initial state
obs, info = env.reset()
print(f"Initial State (Observation): {obs}")

# Initialize episode tracking
done = False
step = 0
total_reward = 0

# Optional: store experience tuples (s, a, r, s', done)
trajectory = []

# Run the episode
while not done:
    action = env.action_space.sample()  # Replace this with agent action later
    next_obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    total_reward += reward

    # Save transition
    trajectory.append((obs, action, reward, next_obs, done))

    # Print details
    print(f"\nStep {step}")
    print(f"  Action Taken: {action}")
    print(f"  Reward: {reward}")
    print(f"  Next Observation: {next_obs}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")

    # Prepare for next step
    obs = next_obs
    step += 1

print("\nEpisode finished.")
print(f"Total Reward Collected: {total_reward}")
print(f"Total Steps: {step}")
