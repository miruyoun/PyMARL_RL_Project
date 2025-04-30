import torch
import torch.nn as nn

class MAPPOAgent(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_dim, device):
        super(MAPPOAgent, self).__init__()
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # <<< Move to device AFTER fully constructing layers
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

    def act(self, obs):
        obs = obs.to(self.device)
        return self.actor(obs)

    def evaluate_value(self, state):
        state = state.to(self.device)
        return self.critic(state)
