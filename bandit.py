import random

class Stationary:
    def __init__(self, k) -> None:
        self.k = k
        self.A = [ a for a in range(self.k) ]
        self.reset()
    
    def __call__(self, a) -> float:
        return random.gauss(mu=self.q[a], sigma=1)
    
    def reset(self):
        self.q = { a: random.gauss(mu=0, sigma=1) for a in self.A }
        self.o = max(self.A, key=self.q.__getitem__)
