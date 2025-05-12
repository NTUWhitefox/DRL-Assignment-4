import gymnasium as gym
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

LOG_STD_MIN = 2
LOG_STD_MAX = -20
EPS = 1e-6
device = ('cpu')

class PolicyNetSAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, action_bound_val=1.0):
        super(PolicyNetSAC, self).__init__()
        self.action_dim = action_dim
        # Assuming action_bound_val means actions are in [-action_bound_val, action_bound_val]
        # For actions in [-1, 1], action_bound_val = 1.0
        self.action_scale = torch.tensor(action_bound_val)

        self.seq = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.seq(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        sample = dist.rsample()
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample) - torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(axis=-1, keepdim=True)

        return action, log_prob
    def get_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0).to(device, dtype=torch.float)
        true_action, _ = self.sample(state)
        return true_action.detach().cpu().numpy()
    
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.actor = PolicyNetSAC(state_dim = 67, action_dim = 21, hidden_size=256)
        #load from policy.pth
        self.actor.load_state_dict(torch.load("policy.pth"))

    def act(self, observation):
        return self.actor.get_action(observation)
