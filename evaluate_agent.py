import gymnasium as gym
import numpy as np
import argparse
import importlib
import torch
from my_policy import PPOActorCritic
import imageio
import matplotlib.pyplot as plt
from datetime import datetime

def evaluate_policy(policy, policy_action, total_episodes=100, render_first=5, save_gif=True):
    total_reward = 0.0
    rewards_history = []
    frames = []
    
    for episode in range(total_episodes):
        # Render the first few episodes
        render_mode = "rgb_array" if episode < render_first else "rgb_array"
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy_action(policy, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Save frames for GIF (only for first episode)
            if episode == 0 and save_gif:
                frames.append(env.render())
                
        env.close()
        total_reward += episode_reward
        rewards_history.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{total_episodes}")
    
    # Save GIF
    if save_gif and frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imageio.mimsave(f'lunar_lander_{timestamp}.gif', frames, fps=30)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Reward History over 100 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'rewards_plot_{timestamp}.png')
    plt.show()
    
    return total_reward / total_episodes, rewards_history

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an AI agent for LunarLander-v3 using a provided policy and policy_action function."
    )
    parser.add_argument(
        "--filename", type=str, required=True,
        help="Path to the .npy file containing the policy parameters."
    )
    parser.add_argument(
        "--policy_module", type=str, required=True,
        help="The name of the Python module that defines the policy_action function."
    )
    args = parser.parse_args()

    # Load the policy parameters from the file.
    state_dim = 8  # LunarLander observation space size
    action_dim = 4  # LunarLander has 4 discrete actions
    policy = PPOActorCritic(state_dim, action_dim)
    policy.load_state_dict(torch.load(args.filename))
    policy.eval()
    
    # Dynamically import the module that defines policy_action.
    try:
        policy_module = importlib.import_module(args.policy_module)
    except ImportError as e:
        print(f"Error importing module {args.policy_module}: {e}")
        return

    # Verify that the module has a callable policy_action function.
    if not hasattr(policy_module, "policy_action") or not callable(policy_module.policy_action):
        print(f"Module {args.policy_module} must define a callable 'policy_action(policy, observation)' function.")
        return
    policy_action_func = policy_module.policy_action

    # Evaluate the policy over 100 episodes (first 5 are rendered).
    average_reward, rewards_history = evaluate_policy(policy, policy_action_func, 
                                                    total_episodes=100, render_first=5)
    print(f"Average reward over 100 episodes: {average_reward:.2f}")
    print(f"Best episode reward: {max(rewards_history):.2f}")
    print(f"Worst episode reward: {min(rewards_history):.2f}")

if __name__ == "__main__":
    main()
