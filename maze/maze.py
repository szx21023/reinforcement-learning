import random
import numpy as np

from const import REWARD, REWARD_OBSTACLE

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
            self.reward_table[tuple(pos)] = REWARD_OBSTACLE
        self.reward_table[tuple(self.reward_point)] = REWARD
    
    def get_obstacle(self) -> list:
        return self.obstacle
    
    def get_reward_point(self) -> list:
        return self.reward_point
    
    def get_reward_table(self) -> np.array:
        return self.reward_table
    
    def out_of_boundary(self, h, w) -> bool:
        return (h < 0) | (h >= self.height) | (w < 0) | (w >= self.width)