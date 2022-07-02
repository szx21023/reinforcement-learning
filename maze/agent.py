import random
import numpy as np

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
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[pos])]

    def __str__(self):
        return f'{self.position}'