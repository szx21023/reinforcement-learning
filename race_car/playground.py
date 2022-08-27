import pandas as pd
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

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def get_coordinate(self):
        return self.x, self.y
    
    def set_coordinate(self):
        pass
    
    def situaction(self, runaway):
        dist_l = 0
        dist_lf = 0
        dist_f = 0
        dist_r = 0
        dist_rf = 0
        
        while (self.x-(dist_l+1) >= 0) and (runaway[self.y][self.x-(dist_l+1)] != 0):
            dist_l += 1
           
        while (self.x-(dist_lf+1) >= 0) and (self.y-(dist_lf+1) >= 0) and (runaway[self.y-(dist_lf+1)][self.x-(dist_lf+1)]) != 0:
            dist_lf += 1
           
        while (self.y-(dist_f+1) >= 0) and (runaway[self.y-(dist_f+1)][self.x] != 0):
            dist_f += 1
          
        while (self.x+(dist_r+1) < self.w) and (runaway[self.y][self.x+(dist_r+1)] != 0):
            dist_r += 1
          
        while (self.x+(dist_rf+1) < self.w) and (self.y-(dist_rf+1) >= 0) and (runaway[self.y-(dist_rf+1)][self.x+(dist_rf+1)] != 0):
            dist_rf += 1
            
        return dist_l, dist_lf, dist_f, dist_r, dist_rf

playground = Playground(10, 15, 4)
playground.runaway