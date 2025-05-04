import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- 離散動作空間 ---
ACTIONS = [
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Left
    np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Right
    np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),   # Brake
    np.array([0.0, 0.0, 0.0], dtype=np.float32),   # No-op
]

# --- 模型定義 ---
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)

# --- 圖像預處理 ---
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)  # (1, 84, 84)

# --- Agent 類別 ---
class DQNAgent:
    def __init__(self, num_actions, device, gamma=0.99, lr=1e-4, pretrain_model_path=None):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.model = DQN(num_actions).to(device)
        if pretrain_model_path:
            self.model.load_state_dict(torch.load(pretrain_model_path, map_location=self.device))
        self.target_model = DQN(num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_interval = 1000
        self.step_count = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (~dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- 主程式 ---
def main():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(num_actions=len(ACTIONS), device=device)

    for episode in range(1000):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0

        for t in range(500):
            action_idx = agent.act(state)
            action = ACTIONS[action_idx]
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(obs)
            done = terminated or truncated

            agent.remember(state, action_idx, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode}, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.2f}")

        if episode % 20 == 0:
            torch.save(agent.model.state_dict(), f"car_dqn_episode{episode}.pth")

    env.close()

if __name__ == "__main__":
    main()
