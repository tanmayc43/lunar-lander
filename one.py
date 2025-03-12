import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

for _ in range(1000):  # Run for 1000 time steps
    action = env.action_space.sample()  # Choose a random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:  # Restart if the episode ends
        observation, info = env.reset()

env.close()
