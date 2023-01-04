from _init_._init_ import *
from _init_.Epsilongreedy import epgreedy
import numpy as np

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    gamma_decay = np.linspace(gamma, 0, max_episodes)
    q = np.zeros(4)
    for i in range(max_episodes):
        features = env.reset()
        for a in env.actions:
            q[a] = features.dot(theta[i])
        #e = np.zeros(env.n_features)
        done  = False
        while not done:
            a = epgreedy(epsilon[i], q)
            features_, r, done = env.step(a)
            delta = r - q
            q_ = np.zeros(4)
            for a_ in env.actions:
                q_[a_] = features_.dot(theta[i])
            delta = delta + gamma_decay[i]*max(q_)
            theta = theta + eta[i]*delta*features
            features = features_


    return theta 