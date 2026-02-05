from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

"""
Network backbones for RL Agents
Available Backbones:
1. Backbone for Vanilla Agent
2. Backbone for action branching

Replay buffers for storing state transitions
1. vanilla replay buffer
2. time series replay buffer
"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# RL Controllers
class DQN_v0(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super(DQN_v0, self).__init__()
        self.fc1 = nn.Linear(in_dim, 25)
        self.fc2 = nn.Linear(25,25)
        self.out = nn.Linear(25, out_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        out = self.out(x2)
        return out

# RL Controller with action branching
class DQN_AB(nn.Module):
    def __init__(self, s_dim=10, h_dim=25, branches=[1,2,3]):
        super(DQN_AB, self).__init__()
        self.s_dim, self.h_dim = s_dim, h_dim
        self.branches = branches
        self.shared = nn.Sequential(nn.Linear(self.s_dim, self.h_dim), nn.ReLU())
        self.shared_state = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU())
        self.domains, self.outputs = [], []
        for i in range(len(branches)):
            layer = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU())
            self.domains.append(layer)
            layer_out = nn.Sequential(nn.Linear(self.h_dim*2, branches[i]))
            self.outputs.append(layer_out)

    def forward(self, x):
        # return list of tensors, each element is Q-Values of a domain
        f = self.shared(x)
        s = self.shared_state(f)
        outputs = []
        for i in range(len(self.branches)):
            branch = self.domains[i](f)
            branch = torch.cat([branch,s],dim=1)
            outputs.append(self.outputs[i](branch))

        return outputs


class QueueBuffer(torch.utils.data.Dataset):
    """
    Basic ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def __getitem__(self, idx):        
        return self.memory[idx] 

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(None)
        if len(self.memory) > self.capacity:
            # to avoid index out of range
            self.memory.pop(0)
        transition = Transition(*args)
        self.memory[-1] = transition


class ReplayMemory(torch.utils.data.Dataset):
    """
    Basic ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __getitem__(self, idx):        
        return self.memory[idx] 

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # to avoid index out of range
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


class ReplayMemoryTime(torch.utils.data.Dataset):
    """
    Time series ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity, w):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.w = w

    def __getitem__(self, idx):        
        return self.memory[idx:idx+self.w] 

    def __len__(self):
        return len(self.memory) - self.w + 1

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # to avoid index out of range
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity