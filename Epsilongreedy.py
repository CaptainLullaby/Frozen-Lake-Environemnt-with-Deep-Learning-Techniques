import random

def epgreedy(epsilon, q):
    p = random.uniform(0,1)
    
    if(p > epsilon):
        return random.randint(0, 3)
    
    else:
        return q.argmax()  