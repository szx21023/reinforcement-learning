import pygame
import sys
import pygame.locals
from maze import Maze
from agent import Agent
from action import ACTIONS
from const import WIDTH, HEIGHT, RECT_SIZE, \
    COLOR_BLACK, COLOR_GREEN, COLOR_NAVYBLUE, COLOR_SEASHELL, \
    REWARD_OBSTACLE

maze = Maze(height=HEIGHT, width=WIDTH, obstacle_ratio=0.1)
maze.set_obstacle()
maze.set_reward_point()
maze.set_reward_table() 
reward_table = maze.get_reward_table()
agent = Agent([0, 0], ACTIONS)
agent.set_q_table(*reward_table.shape)

# 初始化
pygame.init()
# 設定螢幕的大小
screen = pygame.display.set_mode((WIDTH*RECT_SIZE, HEIGHT*RECT_SIZE))
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
        action = agent.get_next_step(agent.position)
        direction = action.direction
        new_position = [agent.position[0] + direction[0], agent.position[1] + direction[1]]
        if maze.out_of_boundary(*new_position):
            reward = REWARD_OBSTACLE
            agent.learn(agent.position, action.code, reward)
            print('game over!')
            break

        reward = reward_table[tuple(new_position)]
        print(agent, action, reward)
        agent.learn(agent.position, action.code, reward, new_position)
        agent.position = new_position
        if maze.reward_point == new_position:
            print('get reward!')
            break

        screen.fill(COLOR_SEASHELL)
        for obs_position in maze.get_obstacle():
            pygame.draw.rect(screen, COLOR_BLACK, [obs_position[1]*RECT_SIZE, obs_position[0]*RECT_SIZE, RECT_SIZE, RECT_SIZE])
        pygame.draw.rect(screen, COLOR_NAVYBLUE, [agent.position[1]*RECT_SIZE, agent.position[0]*RECT_SIZE, RECT_SIZE, RECT_SIZE])
        pygame.draw.rect(screen, COLOR_GREEN, [maze.reward_point[1]*RECT_SIZE, maze.reward_point[0]*RECT_SIZE, RECT_SIZE, RECT_SIZE])
        pygame.draw.lines(screen, COLOR_BLACK, closed=False, points=[[0, 0], [0, HEIGHT*RECT_SIZE], [WIDTH*RECT_SIZE, HEIGHT*RECT_SIZE], [WIDTH*RECT_SIZE, 0], [0, 0]], width=1)
        pygame.display.update()
