import numpy as np
import gymnasium as gym

def evaluate_policy(env, policy, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(np.dot(policy, state))  # Simple linear policy
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    return total_reward / num_episodes  # Average reward

def train():
    env = gym.make("LunarLander-v3")
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize a random policy (simple linear model)
    best_policy = np.random.randn(num_actions, num_features)
    best_reward = -np.inf

    for _ in range(1000):  # Train for 1000 generations
        new_policy = best_policy + np.random.randn(num_actions, num_features) * 0.1
        new_reward = evaluate_policy(env, new_policy)

        if new_reward > best_reward:
            best_reward = new_reward
            best_policy = new_policy

    np.save("best_policy_<group>.npy", best_policy)  # Save the best policy
    env.close()

if __name__ == "__main__":
    train()
