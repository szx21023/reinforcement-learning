import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# --- 離散動作空間（正確型別：np.ndarray） ---
ACTIONS = [
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Left
    np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Right
    np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),   # Brake
    np.array([0.0, 0.0, 0.0], dtype=np.float32),   # No-op
]

# --- CNN 模型 ---
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)

# --- 圖像預處理（灰階 + 縮放 + 正規化）---
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)  # shape: (1, 84, 84)

# --- 初始化環境與模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CarRacing-v3", render_mode="rgb_array")

model = DQN(num_actions=len(ACTIONS)).to(device)
target_model = DQN(num_actions=len(ACTIONS)).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-4)
memory = deque(maxlen=10000)

# --- 訓練參數 ---
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
UPDATE_TARGET = 1000
STEP = 0

# --- 主訓練迴圈 ---
for episode in range(1000):
    obs, _ = env.reset()
    state = preprocess(obs)
    total_reward = 0

    for t in range(500):
        STEP += 1

        # ε-greedy 動作選擇
        if np.random.rand() < EPSILON:
            action_idx = np.random.randint(len(ACTIONS))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action_idx = torch.argmax(q_values).item()

        action = ACTIONS[action_idx]
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess(obs)
        done = terminated or truncated

        memory.append((state, action_idx, reward, next_state, done))
        state = next_state
        total_reward += reward

        # 訓練模型
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # ✅ 使用 np.array() 統一轉 tensor（消除警告 + 加速）
            states = torch.FloatTensor(np.array(states)).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

            q_values = model(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_model(next_states).max(1, keepdim=True)[0]
                target_q = rewards + GAMMA * max_next_q * (~dones)

            loss = nn.functional.mse_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if STEP % UPDATE_TARGET == 0:
            target_model.load_state_dict(model.state_dict())

        if done:
            break

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    print(f"Episode {episode}, reward: {total_reward:.2f}, epsilon: {EPSILON:.2f}")

    # 可選：定期儲存模型
    if episode % 20 == 0:
        torch.save(model.state_dict(), f"car_dqn_episode{episode}.pth")

env.close()
