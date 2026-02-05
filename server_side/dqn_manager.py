from typing import Optional
from time import sleep
import os
import sys
import math

import csv
import json
import time
import datetime
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent_ds import DQN_v0, DQN_AB, ReplayMemory

def convert_tuple_to_tensors(data_tuple):
    """
    Convert a 4-element tuple with mixed int and python list to tensors.
    Each element is converted to either a tensor scalar or torch.tensor.
    If 1D list is provided, it's converted to shape (1, len(list)).
    
    Args:
        data_tuple: tuple of 4 elements (int or list)
    
    Returns:
        tuple of 4 torch tensors
    """
    converted = []
    for item in data_tuple:
        if isinstance(item, (int, float)):
            # Convert scalar to tensor
            tensor = torch.tensor(item)
        elif isinstance(item, list):
            # Convert list to tensor
            if len(item) == 0:
                tensor = torch.tensor([])
            else:
                # Check if it's 1D list
                if not isinstance(item[0], list):
                    # 1D list: reshape to (1, len(list))
                    tensor = torch.tensor(item).unsqueeze(0)
                else:
                    # 2D or higher: convert as is
                    tensor = torch.tensor(item)
        elif isinstance(item, np.ndarray):
            # Convert numpy array to tensor
            if item.ndim == 1:
                tensor = torch.from_numpy(item).unsqueeze(0)
            else:
                tensor = torch.from_numpy(item)
        elif torch.is_tensor(item):
            # Already a tensor
            tensor = item
        else:
            raise TypeError(f"Unsupported type: {type(item)}")
        
        converted.append(tensor)
    
    return tuple(converted)

def save_checkpoint(state, savepath, flag=True):
    """Save for general purpose (e.g., resume training)"""
    if not os.path.isdir(savepath):
        os.makedirs(savepath, 0o777)
    # timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if flag:
        filename = os.path.join(savepath, "best_ckpt.pth.tar")
    else:
        filename = os.path.join(savepath, "newest_ckpt.pth.tar")
    torch.save(state, filename)


def load_checkpoint(savepath, flag=True):
    """Load for general purpose (e.g., resume training)"""
    if flag:
        filename = os.path.join(savepath, "best_ckpt.pth.tar")
    else:
        filename = os.path.join(savepath, "newest_ckpt.pth.tar")
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

# Define DQN_AGENT_AB
class DQN_AGENT_AB():
    def __init__(self, s_dim, h_dim, branches, buffer_size, params):
        self.eps = 0.8
        self.actions = [np.arange(i) for i in branches]
        
        # Replay Memory
        self.mem = ReplayMemory(buffer_size)
        
        # Policy and Target Networks
        self.policy_net = DQN_AB(s_dim, h_dim, branches)
        self.target_net = DQN_AB(s_dim, h_dim, branches)
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss function
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
        
    def max_action(self, state):
        # actions for multidomains
        max_actions = []
        with torch.no_grad():
            # Inference using policy_net given (domain, batch, dim)
            q_values = self.policy_net(state)
            for i in range(len(q_values)):
                domain = q_values[i].max(dim=1).indices
                max_actions.append(self.actions[i][domain])
        return max_actions
    
    def e_gready_action(self, actions, eps):
        # Epsilon-Gready for exploration
        final_actions = []
        for i in range(len(actions)):
            p = np.random.random()
            if isinstance(actions[i],np.ndarray):
                if p < 1- eps:
                    final_actions.append(actions[i])
                else:
                    # randint in (0, domain_num), for batchsize
                    final_actions.append(np.random.randint(len(self.actions[i]),size=len(actions[i])))
            else:
                if p < 1- eps:
                    final_actions.append(actions[i])
                else:
                    final_actions.append(np.random.choice(self.actions[i]))
        final_actions = [int(i) for i in final_actions]
        return final_actions
    
    def select_action(self, state):
        return self.e_gready_action(self.max_action(state),self.eps)
    
    def train(self, n_round, n_update, n_batch):
        # Train on policy_net
        losses = []
        self.target_net.train()
        train_loader = torch.utils.data.DataLoader(
            self.mem, batch_size=n_batch, shuffle=True, drop_last=True)
        length = len(train_loader.dataset)
        GAMMA = 1.0

        # Calcuate loss for each branch and then simply sum up
        for i, trans in enumerate(train_loader):
            if i > n_update:
                break
            
            loss = 0.0 # initialize loss at the beginning of each batch
            states, actions, next_states, rewards = trans
            with torch.no_grad():
                target_result = self.target_net(next_states)
            policy_result = self.policy_net(states)
            # Loop through each action domain
            for j in range(len(self.actions)):
                next_state_values = target_result[j].max(dim=1)[0].detach()
                expected_state_action_values = (next_state_values*GAMMA) + rewards.float()
                # Gather action-values that have been taken
                branch_actions = actions[j].long() # branch_actions = actions[j].long()
                state_action_values = policy_result[j].gather(1, branch_actions.unsqueeze(1))
                loss += self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return losses
    
    def save_model(self, n_round, savepath):
        save_checkpoint({'epoch': n_round, 'model_state_dict':self.target_net.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()}, savepath)
        f = open(os.path.join(savepath,"memory"), 'wb')
        pickle.dump(self.mem,f)
        f.close()

    def load_model(self, loadpath):
        if not os.path.isdir(loadpath): os.makedirs(loadpath)
        checkpoint = load_checkpoint(loadpath)
        if checkpoint is not None:
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.eval()
        if os.path.exists(os.path.join(loadpath,"memory")):
            try:
                f = open(os.path.join(loadpath,"memory"),'rb')
                self.mem = pickle.load(f)
                f.close()
                print(f"Successfully loaded memory with {len(self.mem)} samples")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Failed to load memory file ({e}). Using empty memory buffer.")
                # Keep the original empty memory initialized in __init__

    def sync_model(self):
	    self.target_net.load_state_dict(self.policy_net.state_dict())