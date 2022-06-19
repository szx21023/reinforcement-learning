import numpy as np
import random
import pygame
import sys
import pygame.locals

class Action:
    def __init__(self, name: str, code: int, direction: list):
        self.name = name
        self.code = code
        self.direction = direction
        
ACTIONS = [
    Action('up', 0, [-1, 0]),
    Action('down', 1, [1, 0]),
    Action('left', 2, [0, -1]),
    Action('right', 3, [0, 1])
]

class Maze:
    def __init__(self, height: int, width: int, obstacle_ratio: float):
        self.height = height
        self.width = width
        self.obstacle_ratio = obstacle_ratio
        self.array = [[h, w] for h in range(height) for w in range(width)]
        self.table = np.zeros(shape=(height, width))
    
    def set_obstacle(self):
        num = int((self.height*self.width)*self.obstacle_ratio)
        self.obstacle = random.sample(self.array, num)
        for pos in self.obstacle:
            self.array.remove(pos)
    
    def set_reward_point(self):
        self.reward_point = random.sample(self.array, 1)[0]
        self.array.remove(self.reward_point)
        
    def set_reward_table(self):
        self.reward_table = np.array([[-1 for w in range(self.width)] for h in range(self.height)])
        for pos in self.obstacle:
            self.reward_table[tuple(pos)] = -50
        self.reward_table[tuple(self.reward_point)] = 200
    
    def get_obstacle(self) -> list:
        return self.obstacle
    
    def get_reward_point(self) -> list:
        return self.reward_point
    
    def get_reward_table(self) -> np.array:
        return self.reward_table
    
    def out_of_boundary(self, h, w) -> bool:
        return (h < 0) | (h >= self.height) | (w < 0) | (w >= self.width)
    
class Agent:
    def __init__(self, position: list, actions: list, alpha: float=0.1, beta: float=0.2, gamma: float=0.1):
        self.position = position
        self.actions = actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def set_q_table(self, height: int, width: int):
        self.q_table = np.zeros(shape=(height, width, len(self.actions)))
    
    def learn(self, pos: list, a: int, reward: int, new_pos: list=[]):
        pos = tuple(pos)
        new_pos = tuple(new_pos)
        if new_pos == ():
            self.q_table[pos][a] = self.q_table[pos][a] + self.alpha * \
            (reward - self.q_table[pos][a])
        else:
            self.q_table[pos][a] = self.q_table[pos][a] + self.alpha * \
            (reward + self.beta * max(self.q_table[new_pos]) - self.q_table[pos][a])

    def get_next_step(self, pos: list) -> int:
        pos = tuple(pos)
        if self.gamma > random.random():
            return random.choice(self.actions).code
        else:
            return np.argmax(agent.q_table[pos])

width =20
height = 20

maze = Maze(height=height, width=width, obstacle_ratio=0.1)
maze.set_obstacle()
maze.set_reward_point()
maze.set_reward_table() 
reward_table = maze.get_reward_table()
agent = Agent([0, 0], ACTIONS)
agent.set_q_table(*reward_table.shape)

rect_size = 20
COLOR_SEASHELL = 255, 245, 238
COLOR_NAVYBLUE = 0, 0, 191
COLOR_GREEN = 0, 255, 0
COLOR_BLACK = 0, 0, 0

# 初始化
pygame.init()
# 設定螢幕的大小
screen = pygame.display.set_mode((width*rect_size, height*rect_size))
# 設定螢幕的名稱
pygame.display.set_caption("Maze")

clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            # 如果點選關閉按鈕，或者按下任意鍵，那麼退出程式
            sys.exit()

    screen.fill(COLOR_SEASHELL)

    agent.position = [0, 0]
    while True:
        clock.tick(20)
        a_index = agent.get_next_step(agent.position)
        direction = ACTIONS[a_index].direction
        new_position = [agent.position[0] + direction[0], agent.position[1] + direction[1]]
        if maze.out_of_boundary(*new_position):
            reward = -50
            agent.learn(agent.position, a_index, reward)
            print('game over!')
            break

        reward = reward_table[tuple(new_position)]
        print(agent.position, a_index, reward)
        agent.learn(agent.position, a_index, reward, new_position)
        agent.position = new_position
        if maze.reward_point == new_position:
            print('get reward!')
            break

        screen.fill(COLOR_SEASHELL)
        for obs_position in maze.get_obstacle():
            pygame.draw.rect(screen, COLOR_BLACK, [obs_position[1]*rect_size, obs_position[0]*rect_size, rect_size, rect_size])
        pygame.draw.rect(screen, COLOR_NAVYBLUE, [agent.position[1]*rect_size, agent.position[0]*rect_size, rect_size, rect_size])
        pygame.draw.rect(screen, COLOR_GREEN, [maze.reward_point[1]*rect_size, maze.reward_point[0]*rect_size, rect_size, rect_size])
        pygame.draw.lines(screen, COLOR_BLACK, closed=False, points=[[0, 0], [0, height*rect_size], [width*rect_size, height*rect_size], [width*rect_size, 0], [0, 0]], width=1)
        pygame.display.update()
