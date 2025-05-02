import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# --- DQN 模型 ---
class DQNCNN(nn.Module):
    def __init__(self, output_dim):
        super(DQNCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

# --- 訓練邏輯 ---
class Agent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.memory = deque(maxlen=1000)
        self.model = DQNCNN(11, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state)
        move = torch.argmax(prediction).item()
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]][move]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_t = torch.tensor(state, dtype=torch.float)
            next_state_t = torch.tensor(next_state, dtype=torch.float)
            action_index = [1, 0, 0].index(action)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_t)).item()
            output = self.model(state_t)
            target_f = output.clone().detach()
            target_f[action_index] = target
            loss = self.loss_fn(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay