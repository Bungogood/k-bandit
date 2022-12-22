import random
import math

class Simple:
    def __init__(self, k, e) -> None:
        self.A = [ a for a in range(k) ]
        self.e = e
        self.reset()
    
    def __str__(self) -> str:
        return 'simple (e={})'.format(self.e)
    
    def reset(self):
        self.Q = { a: 0 for a in self.A }
        self.N = { a: 0 for a in self.A }

    def select(self):
        if random.random() > self.e:
            return max(self.A, key=self.Q.__getitem__)
        else:
            return random.choice(self.A)
    
    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += 1 / self.N[a] * (r - self.Q[a])

class Tracking:
    def __init__(self, k, e, s) -> None:
        self.A = [ a for a in range(k) ]
        self.e = e
        self.s = s
    
    def __str__(self) -> str:
        return 'tracking (e={}, a={})'.format(self.e, self.s)
    
    def reset(self):
        self.Q = { a: 0 for a in self.A }

    def select(self):
        if random.random() > self.e:
            return max(self.A, key=self.Q)
        else:
            return random.choice(self.A)
    
    def update(self, a, r):
        self.Q[a] += self.s * (r - self.Q[a])

class Optimist:
    def __init__(self, k, e, q0) -> None:
        self.A = [ a for a in range(k) ]
        self.e = e
        self.q0 = q0
        self.reset()
    
    def __str__(self) -> str:
        return 'optimistic (e={}, q0={})'.format(self.e, self.q0)
    
    def reset(self):
        self.Q = { a: self.q0 for a in self.A }
        self.N = { a: 0 for a in self.A }

    def select(self):
        if random.random() > self.e:
            return max(self.A, key=self.Q.__getitem__)
        else:
            return random.choice(self.A)
    
    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += 1 / self.N[a] * (r - self.Q[a])

class UCB:
    def __init__(self, k, c) -> None:
        self.A = [ a for a in range(k) ]
        self.c = c
        self.reset()
    
    def __str__(self) -> str:
        return 'UCB (c={})'.format(self.c)
    
    def reset(self):
        self.Q = { a: 0 for a in self.A }
        self.N = { a: 0 for a in self.A }
        self.t = 0
    
    def eval(self, a):
        # when self.N[a] == 0 then it is maximising
        if self.N[a] == 0: return self.Q[a] + 10000
        return self.Q[a] + self.c * math.sqrt(math.log(self.t+1) / self.N[a])

    def select(self):
        return max(self.A, key=self.eval)
    
    def update(self, a, r):
        self.t += 1
        self.N[a] += 1
        self.Q[a] += 1 / self.N[a] * (r - self.Q[a])



class Gibbs:
    def __init__(self, k, c) -> None:
        self.A = [ a for a in range(k) ]
        self.c = c
        self.reset()
    
    def __str__(self) -> str:
        return 'Gibbs (c={})'.format(self.c)
    
    def reset(self):
        self.H = { a: 0 for a in self.A }
        self.R = 0
        self.t = 0
    
    def prob(self):
        d = sum(math.exp(self.H[a]) for a in self.A)
        return { a: math.exp(self.H[a]) / d for a in self.A }

    def select(self):
        return max(self.A, key=self.H.__getitem__)
    
    def update(self, a, r):
        pi = self.prob()
        for b in self.A:
            if b == a:
                self.H[b] += self.c * (r - self.R) * (1 - pi[b])
            else:
                self.H[b] -= self.c * (r - self.R) * pi[b]
        self.t += 1
        self.R += (1 / self.t) * (r - self.R)
