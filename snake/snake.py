from const import DIRECT_UP, DIRECT_DOWN, DIRECT_LEFT, DIRECT_RIGHT

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

    def __str__(self):
        return f'{self.body}'