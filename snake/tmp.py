import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import time
import cv2

# 遊戲設定
WIDTH, HEIGHT = 40, 40
BLOCK_SIZE = 5
SPEED = 20
ACTIONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# CNN 模型
class DQNCNN(nn.Module):
    def __init__(self, output_dim):
        super(DQNCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2),  # 2 channels: gray + border
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Snake 遊戲環境
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DQN Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        self.reset()

    def reset(self):
        x = WIDTH // 2
        y = HEIGHT // 2
        self.head = [x, y]
        self.snake = [[x, y], [x - BLOCK_SIZE, y]]
        self.direction = (BLOCK_SIZE, 0)
        self.food = self._place_food()
        self.frame = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if [x, y] not in self.snake:
                return [x, y]

    def _is_collision(self):
        x, y = self.head
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _draw(self, score):
        self.display.fill((0, 0, 0))

        for i, s in enumerate(self.snake):
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            pygame.draw.rect(self.display, color, pygame.Rect(s[0], s[1], BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, (255, 255, 0), pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        score_text = self.font.render(f"Score: {score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(SPEED)

    def _get_state(self):
        raw_pixels = pygame.surfarray.array3d(self.display)
        gray = np.mean(raw_pixels, axis=2)
        resized = cv2.resize(gray, (84, 84)) / 255.0

        border = np.zeros((84, 84), dtype=np.float32)
        border[0, :] = 1
        border[-1, :] = 1
        border[:, 0] = 1
        border[:, -1] = 1

        state = np.stack([resized, border], axis=0)
        return state.astype(np.float32)

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        clock_wise = [(BLOCK_SIZE, 0), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (0, -BLOCK_SIZE)]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = clock_wise[(idx + 1) % 4]
        self.direction = new_dir

        new_head = [self.head[0] + self.direction[0], self.head[1] + self.direction[1]]
        self.head = new_head
        self.snake.insert(0, new_head)

        reward = 0
        done = False
        if self.head == self.food:
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        if self._is_collision():
            done = True
            reward = -10

        self.frame += 1
        self._draw(score=len(self.snake) - 2)
        return self._get_state(), reward, done

# Agent
class Agent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.memory = deque(maxlen=1000)
        self.model = DQNCNN(3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state)
        move = torch.argmax(prediction).item()
        return ACTIONS[move]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            next_state_t = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            action_index = ACTIONS.index(action)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_t)).item()

            output = self.model(state_t)
            target_f = output.clone().detach()
            target_f[0][action_index] = target

            loss = self.loss_fn(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay

# 訓練主程式
if __name__ == "__main__":
    env = SnakeGame()
    agent = Agent()
    episodes = 1000

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(64)
            state = next_state
            total_reward += reward

            time.sleep(0.05)

            if done:
                print(f"Episode {ep}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break