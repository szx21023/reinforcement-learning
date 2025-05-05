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
class HybridDQN(nn.Module):
    def __init__(self, num_actions):
        super(HybridDQN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, image, speed):
        x = self.cnn(image)
        x = torch.cat([x, speed], dim=1)
        return self.fc(x)

def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)  # (1, 84, 84)

class DQNAgent:
    def __init__(self, num_actions, device, gamma=0.99, lr=1e-4, pretrain_model_path=None):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.model = HybridDQN(num_actions).to(device)
        if pretrain_model_path:
            self.model.load_state_dict(torch.load(pretrain_model_path, map_location=device))
        self.target_model = HybridDQN(num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_interval = 1000
        self.step_count = 0

    def act(self, state, speed):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        speed_tensor = torch.FloatTensor([[speed]]).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor, speed_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, speed, action, reward, next_state, next_speed, done):
        self.memory.append((state, speed, action, reward, next_state, next_speed, done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, speeds, actions, rewards, next_states, next_speeds, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        speeds = torch.FloatTensor(np.array(speeds)).unsqueeze(1).to(self.device)
        next_speeds = torch.FloatTensor(np.array(next_speeds)).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states, speeds).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_model(next_states, next_speeds).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (~dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def main():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(num_actions=len(ACTIONS), device=device)

    for episode in range(1000):
        obs, _ = env.reset()
        state = preprocess(obs)
        speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
        total_reward = 0

        for t in range(10000):
            action_idx = agent.act(state, speed)
            action = ACTIONS[action_idx]
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(obs)
            next_speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
            done = terminated

            agent.remember(state, speed, action_idx, reward, next_state, next_speed, done)
            agent.train()

            state = next_state
            speed = next_speed
            total_reward += reward

            if done or total_reward < -50:
                break

        print(f"Episode {episode}, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.2f}")
        agent.update_epsilon()

        if episode % 20 == 0:
            torch.save(agent.model.state_dict(), f"car_dqn_episode{episode}.pth")

    env.close()

if __name__ == "__main__":
    main()
