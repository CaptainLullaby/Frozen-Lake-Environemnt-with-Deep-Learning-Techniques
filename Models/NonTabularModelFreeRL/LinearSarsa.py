from _init_._init_ import *
from _init_.Epsilongreedy import *
import numpy as np

def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    gamma_decay = np.linspace(gamma, 0, max_episodes)
    
    
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        a = epgreedy(epsilon[i], q)
        e = np.zeros(env.n_features)
        done  = False
        
        while not done:
            features_, r, done = env.step(a)
            a_ = epgreedy(epsilon[i], q)
            e = e + features[a]
            theta = theta + eta[i] * (r + gamma * q[a_] - q[a]) * e 
            e = gamma_decay[i] * e + features[a]
            a = a_
            features = features_

    
    return theta