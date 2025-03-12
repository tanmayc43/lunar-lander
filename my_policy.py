import numpy as np
import gymnasium as gym

class LunarLanderPolicy:
    def __init__(self, policy_file):
        self.policy = np.load(policy_file)

    def get_action(self, state):
        return np.argmax(np.dot(self.policy, state))  # Choose best action

def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    policy = LunarLanderPolicy("best_policy.npy")

    for _ in range(5):  # Run 5 episodes
        state, _ = env.reset()
        done = False
        while not done:
            action = policy.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()
