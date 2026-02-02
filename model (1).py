import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """Here search : Multi Layer Perceptron, what does it do -- essentially matrix multiplication to bring it to larger dimension"""
    def __init__(self, state_size, action_size, hidden_dim=64):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)  
        )

    def forward(self, x):
        return self.mlp(x)