import random
import pygame
import sys
import pygame.locals

COLOR_SEASHELL = 255, 245, 238
COLOR_NAVYBLUE = 0, 0, 191
COLOR_BLACK = 0, 0, 0

DIRECT_UP = 0
DIRECT_DOWN = 1
DIRECT_LEFT = 2
DIRECT_RIGHT = 3

class Snake:
    def __init__(self, x, y):
        self.head = [x, y]
        self.body = [[x, y]]
        self.direction = DIRECT_UP
        self.reward_state = 0

    def set_direction(self, direction):
        self.direction = direction

    def move(self):
        if self.direction == DIRECT_UP:
            self.head[1] -= 1

        elif self.direction == DIRECT_DOWN:
            self.head[1] += 1

        elif self.direction == DIRECT_LEFT:
            self.head[0] -= 1

        elif self.direction == DIRECT_RIGHT:
            self.head[0] += 1

        # 如果是得到獎勵的狀態，就可以成長了
        if self.reward_state == 1:
            self.reward_state = 0

        else:
            self.body.pop()

        self.body.insert(0, self.head.copy())

    def grow_up(self):
        self.reward_state = 1

class Playground:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def is_in_the_place(self, x, y):
        return (x >= 0) and (x < self.width) and (y >= 0) and (y < self.height)


width = 50
height = 50
rect_size = 10

pygame.init()
# 初始化
screen = pygame.display.set_mode((width*rect_size, height*rect_size))
# 設定螢幕的大小
pygame.display.set_caption("Snake")
# 設定螢幕的名稱

snake = Snake(25, 25)
playground = Playground(width, height)

direction = random.randint(0, 3)
reward = [random.randint(0, width-1), random.randint(0, height-1)]
score = 0
clock = pygame.time.Clock()
while True:
    clock.tick(10)
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            # 如果點選關閉按鈕，或者按下任意鍵，那麼退出程式
            sys.exit()

        if event.type == pygame.locals.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = DIRECT_UP

            elif event.key == pygame.K_DOWN:
                direction = DIRECT_DOWN

            elif event.key == pygame.K_LEFT:
                direction = DIRECT_LEFT

            elif event.key == pygame.K_RIGHT:
                direction = DIRECT_RIGHT

            else:
                pass

    screen.fill(COLOR_SEASHELL)
    print(snake.body, reward)
    snake.set_direction(direction)
    snake.move()
    if not playground.is_in_the_place(*snake.head):
        print("game over")
        break

    if snake.head == reward:
        score += 1
        snake.grow_up()
        print("grow up!")
        reward = [random.randint(0, width-1), random.randint(0, height-1)]

    for coord in snake.body:
        pygame.draw.rect(screen, COLOR_NAVYBLUE, [coord[0]*rect_size, coord[1]*rect_size, rect_size, rect_size])
    pygame.draw.rect(screen, COLOR_NAVYBLUE, [reward[0]*rect_size, reward[1]*rect_size, rect_size, rect_size])
    pygame.draw.lines(screen, COLOR_BLACK, closed=False, points=[[0, 0], [0, height*rect_size], [width*rect_size, height*rect_size], [width*rect_size, 0], [0, 0]], width=1)
    # score board
    font = pygame.font.SysFont(f"score: {score}", 5 * rect_size)
    img = font.render(f"score: {score}", True, COLOR_NAVYBLUE)
    screen.blit(img, (0, height*rect_size*90//100))
    pygame.display.update()
