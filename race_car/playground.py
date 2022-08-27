import numpy as np
import random

class Playground:
    def __init__(self, w, h, size):
        self.w = w
        self.h = h
        self.x = w // 2
        self.size = size
        self.runaway = self.init_runaway()

    def generate_boundary_of_runway(self):
        if self.x - self.size // 2 <= 0:
            r = random.randint(0, 1)

        elif self.x + self.size // 2 >= self.w - 1:
            r = random.randint(-1, 0)

        else:
            r = random.randint(-1, 1)

        left = self.x - self.size // 2 + r
        right = self.x + self.size // 2 + r
        self.x = (left + right) // 2
        return left, right

    def generate_new_runaway(self):
        self.runaway[1:] = self.runaway[:-1]
        left, right = self.generate_boundary_of_runway()
        self.runaway[0] = 0
        self.runaway[0, left:right] = 1

    def get_runaway(self):
        pass

    def init_runaway(self):
        array = np.zeros((self.h, self.w))
        for i in range(self.h):
            left, right = self.generate_boundary_of_runway()
            array[-(i+1)][left: right] = 1
        return array

    def is_in_the_boundary(self, x, y):
        return self.runaway[y][x] == 1