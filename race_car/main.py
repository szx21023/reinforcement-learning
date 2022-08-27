import numpy as np
from playground import Playground

import pygame
import sys
import pygame.locals
from const import HEIGHT, WIDTH, RECT_SIZE, COLOR_BLACK, COLOR_GREEN, COLOR_NAVYBLUE, COLOR_SEASHELL

playground = Playground(WIDTH, HEIGHT, 6)

# 初始化
pygame.init()
# 設定螢幕的大小
screen = pygame.display.set_mode((WIDTH*RECT_SIZE, HEIGHT*RECT_SIZE))
# 設定螢幕的名稱
pygame.display.set_caption("Race_car")

clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            # 如果點選關閉按鈕，或者按下任意鍵，那麼退出程式
            sys.exit()

    screen.fill(COLOR_SEASHELL)

    while True:
        clock.tick(20)
        array = playground.runaway

        screen.fill(COLOR_SEASHELL)
        for y, x in zip(np.where(array==1)[0], np.where(array==1)[1]):
            print(y, x)
            pygame.draw.rect(screen, COLOR_GREEN, [x*RECT_SIZE, y*RECT_SIZE, RECT_SIZE, RECT_SIZE])

        for y, x in zip(np.where(array==1)[0], np.where(array==1)[1]):
            print(y, x)
            pygame.draw.rect(screen, COLOR_BLACK, [x*RECT_SIZE, y*RECT_SIZE, RECT_SIZE, RECT_SIZE])


        pygame.draw.lines(screen, COLOR_BLACK, closed=False, points=[[0, 0], [0, HEIGHT*RECT_SIZE], [WIDTH*RECT_SIZE, HEIGHT*RECT_SIZE], [WIDTH*RECT_SIZE, 0], [0, 0]], width=1)
        pygame.display.update()
        playground.generate_new_runaway()
