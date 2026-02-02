import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    """ Initialize Deep Q Agent"""
    def __init__(self, state_size, action_size, model, gamma=0.95, epsilon=0.8, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_buffer = deque(maxlen=2000)

        ## BELOW can ignore -- this is just configuration settings no need learn ( this is structured )
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def save(self, state, action, reward, next_state, done):
        "Saving each time step"
        self.memory_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy policy for action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0) # shape: (1, state_size)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size=32):
        """Train the model from a random batch of experiences"""
        if len(self.memory_buffer) < batch_size:
            return

        minibatch = random.sample(self.memory_buffer, batch_size)

        ### BELOW is creating empty strucutre to fill with data later 
        states      = np.array([exp[0] for exp in minibatch])
        actions     = np.array([exp[1] for exp in minibatch])
        rewards     = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        done        = np.array([exp[4] for exp in minibatch])

        ### dont worry too much about this, this is configuration related too 
        states_tensor      = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        actions_tensor     = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor     = torch.FloatTensor(rewards)
        done_tensor        = torch.BoolTensor(done)

        # Q(s, a)
        q_values = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)

        # Q(s', a') (next)
        with torch.no_grad():
            max_q_next = self.model(next_states_tensor).max(1)[0]

        # Target
        targets = rewards_tensor + self.gamma * max_q_next
        targets[done_tensor] = rewards_tensor[done_tensor]

        #### BELOW NO NEED WORRY -- FIX STRUCTURED CODE 
        # Loss
        loss = self.criterion(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay