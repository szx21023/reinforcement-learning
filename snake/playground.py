import random
import pygame
from const import COLOR_SEASHELL, COLOR_BLACK, COLOR_NAVYBLUE
from snake import Snake

class Playground:
    def __init__(self, width, height, rect_size):
        self.width = width
        self.height = height
        self.rect_size = rect_size

        self.snake = Snake(width//2, height//2)
        self.snake.direction = random.randint(0, 3)

        self.reward = [random.randint(0, width-1), random.randint(0, height-1)]
        self.score = 0
        self.screen = self.init_pygame()

    def is_in_the_boundary(self):
        x = self.snake.head[0]
        y = self.snake.head[1]
        return (x >= 0) and (x < self.width) and (y >= 0) and (y < self.height)

    def init_pygame(self):
        # 初始化
        pygame.init()
        # 設定螢幕的大小
        screen = pygame.display.set_mode((self.width*self.rect_size, self.height*self.rect_size))
        # 設定螢幕的名稱
        pygame.display.set_caption("Snake")
        return screen

    def reset_reward_position(self):
        while True:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if [x, y] not in self.snake.body:
                break

        self.reward = [x, y]

    def _draw(self):
        self.screen.fill(COLOR_SEASHELL)
        # draw the snake and the reward
        for coord in self.snake.body:
            pygame.draw.rect(self.screen, COLOR_NAVYBLUE, [coord[0]*self.rect_size, coord[1]*self.rect_size, self.rect_size, self.rect_size])
        pygame.draw.rect(self.screen, COLOR_NAVYBLUE, [self.reward[0]*self.rect_size, self.reward[1]*self.rect_size, self.rect_size, self.rect_size])
        pygame.draw.lines(self.screen, COLOR_BLACK, closed=False, points=[[0, 0], [0, self.height*self.rect_size], [self.width*self.rect_size, self.height*self.rect_size], [self.width*self.rect_size, 0], [0, 0]], width=1)
        # score board
        font = pygame.font.SysFont(f"score: {self.score}", 5 * self.rect_size)
        img = font.render(f"score: {self.score}", True, COLOR_NAVYBLUE)
        self.screen.blit(img, (0, self.height*self.rect_size*90//100))
        pygame.display.update()