import random
import numpy as np

def epgreedy(epsilon, q):
    
    if(random.uniform(0,1) > epsilon):
        return random.randint(0, 3)
    
    else:
        return np.random.choice(np.flatnonzero(q == q.max()))