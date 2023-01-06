import random
import numpy as np

def epgreedy(epsilon, q):
    p = random.uniform(0,1)
    
    if(p > epsilon):
        return random.randint(0, 3)
    
    else:
        max_a = np.array(np.where(q == q.max())).flatten()
        return random.choice(max_a)