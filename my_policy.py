import torch
import torch.nn as nn

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

# Define `policy_action` function that the evaluator expects
def policy_action(policy, observation):
    state_tensor = torch.FloatTensor(observation).unsqueeze(0)
    logits, _ = policy(state_tensor)
    action = torch.argmax(logits).item()  # Select the action with the highest probability
    return action
