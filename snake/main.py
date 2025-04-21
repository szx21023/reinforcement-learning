import random
import pygame
import sys
import pygame.locals

from const import DIRECT_UP, DIRECT_DOWN, DIRECT_LEFT, DIRECT_RIGHT
from playground import Playground

width = 50
height = 50
rect_size = 10

playground = Playground(width, height, rect_size)
clock = pygame.time.Clock()
while True:
    clock.tick(10)
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            # 如果點選關閉按鈕，或者按下任意鍵，那麼退出程式
            sys.exit()

        if event.type == pygame.locals.KEYDOWN:
            if event.key == pygame.K_UP:
                playground.snake.set_direction(DIRECT_UP)

            elif event.key == pygame.K_DOWN:
                playground.snake.set_direction(DIRECT_DOWN)

            elif event.key == pygame.K_LEFT:
                playground.snake.set_direction(DIRECT_LEFT)

            elif event.key == pygame.K_RIGHT:
                playground.snake.set_direction(DIRECT_RIGHT)

            else:
                pass

    print(playground.snake, playground.reward)
    playground.snake.move()
    if not playground.is_in_the_boundary():
        print("game over")
        break

    if playground.snake.head == playground.reward:
        playground.score += 1
        playground.snake.grow_up()
        print("grow up!")
        playground.reset_reward_position()

    playground._draw()