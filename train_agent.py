import gymnasium as gym
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
NUM_EPISODES_DEFAULT = 50000
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 4       # Number of epochs per PPO update
BATCH_SIZE = 64         # Minibatch size for PPO updates
ENTROPY_BONUS = 0.01    # Encourages exploration
TARGET_KL = 0.01        # KL divergence target to stop training early

# Define the Actor-Critic Network
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PPOActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value

# Compute discounted returns
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# PPO Policy Update
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, clip_epsilon, update_epochs):
    # Convert the list of states to a numpy array first, then to a tensor
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)

    for _ in range(update_epochs):
        logits, values = model(states)
        values = values.squeeze(1)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO Objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(values, returns)

        # Total loss with entropy bonus
        loss = actor_loss + 0.5 * critic_loss - ENTROPY_BONUS * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Train PPO for a specified number of episodes
def train_ppo(num_episodes=NUM_EPISODES_DEFAULT, filename="best_policy.npy"):
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = PPOActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            states, actions, rewards, log_probs = [], [], [], []
            episode_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, _ = model(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state
                episode_reward += reward
            
            # Compute returns and advantages
            returns = compute_returns(rewards, GAMMA)
            with torch.no_grad():
                # Convert list of states to a numpy array first
                _, values = model(torch.FloatTensor(np.array(states)))
                values = values.squeeze(1)
            advantages = returns - values.numpy()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Perform PPO update
            ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, CLIP_EPSILON, UPDATE_EPOCHS)

            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    # Convert state_dict tensors to numpy arrays and save as .npy file
    model_state = model.state_dict()
    model_numpy = {k: v.cpu().numpy() for k, v in model_state.items()}
    np.save(filename, model_numpy)
    print(f"Model saved to {filename}")
    return model

# Play PPO Trained Model
def play_ppo(filename, episodes=5):
    env = gym.make('LunarLander-v3', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PPOActorCritic(state_dim, action_dim)
    
    # Load the .npy file and convert back to tensors
    model_numpy = np.load(filename, allow_pickle=True).item()
    model_state = {k: torch.tensor(v) for k, v in model_numpy.items()}
    model.load_state_dict(model_state)
    model.eval()
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play PPO agent for LunarLander-v3 using a .npy file.")
    parser.add_argument("--train", action="store_true", help="Train the agent using PPO.")
    parser.add_argument("--play", action="store_true", help="Load the trained model and play.")
    parser.add_argument("--filename", type=str, default="best_policy.npy", help="Filename to save/load the model.")
    parser.add_argument("--num_episodes", type=int, default=50000, help="Number of training episodes.")
    args = parser.parse_args()

    if args.train:
        train_ppo(num_episodes=args.num_episodes, filename=args.filename)
    elif args.play:
        play_ppo(args.filename, episodes=5)
    else:
        print("Please specify --train to train or --play to run the trained agent.")
