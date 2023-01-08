from _init_.Epsilongreedy import *
from _init_._init_ import *
import numpy as np

def sarsa(env, max_episodes, eta, gamma, epsilon, plot = False, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    returns = []
    s = env.reset()
    disc_ret = 0
    
    for i in range(max_episodes):
        s = env.reset()
        a = epgreedy(epsilon[i], q[s])
        done = False
        cde = 0
        
        while not done:
            s_, r, done = env.step(a)
            a_ = epgreedy(epsilon[i], q[s_])
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[s_][a_]) - q[s][a])
            s = s_
            a = a_
            cde += 1
            disc_ret += (gamma**(cde - 1))*r
            
        returns.append(disc_ret)
     
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    
    if plot:
        return returns
    else:     
        return policy, value