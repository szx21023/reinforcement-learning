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