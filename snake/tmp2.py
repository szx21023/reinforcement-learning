import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import time

# --- 遊戲設定 ---
WIDTH, HEIGHT = 40, 40
BLOCK_SIZE = 4
SPEED = 20
ACTIONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# --- DQN 模型 ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- Snake 遊戲環境 ---
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DQN Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        self.reset()

    def reset(self):
        self.head = [WIDTH // 2, HEIGHT // 2]
        self.snake = [self.head[:], [self.head[0] - BLOCK_SIZE, self.head[1]]]
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
        self.display.fill((0, 0, 0))  # 背景黑

        # 畫蛇
        for i, s in enumerate(self.snake):
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            pygame.draw.rect(self.display, color, pygame.Rect(s[0], s[1], BLOCK_SIZE, BLOCK_SIZE))

        # 畫食物
        pygame.draw.rect(self.display, (255, 255, 0), pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        # 分數顯示
        score_text = self.font.render(f"Score: {score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(SPEED)

    def _get_state(self):
        head = self.head
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = self.direction == (-BLOCK_SIZE, 0)
        dir_r = self.direction == (BLOCK_SIZE, 0)
        dir_u = self.direction == (0, -BLOCK_SIZE)
        dir_d = self.direction == (0, BLOCK_SIZE)

        danger_straight = self._is_collision()
        danger_right = self._is_collision() if dir_u else False
        danger_left = self._is_collision() if dir_d else False

        food_dir = [
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down
        ]

        state = [
            danger_straight, danger_right, danger_left,
            dir_l, dir_r, dir_u, dir_d,
            *food_dir
        ]
        return np.array(state, dtype=int)

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

        self.head[0] += self.direction[0]
        self.head[1] += self.direction[1]
        self.snake.insert(0, self.head[:])

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

# --- 訓練邏輯 ---
class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.memory = deque(maxlen=1000)
        self.model = DQN(11, 3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
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
            state_t = torch.tensor(state, dtype=torch.float).to(self.device)
            next_state_t = torch.tensor(next_state, dtype=torch.float).to(self.device)
            action_index = ACTIONS.index(action)

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

# --- 主訓練迴圈 ---
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

            if done:
                print(f"Episode {ep}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break