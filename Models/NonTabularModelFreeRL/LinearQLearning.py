from _init_.Epsilongreedy import epgreedy
import numpy as np
import random

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, plot = False, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    gamma_decay = np.linspace(gamma, 0, max_episodes)

    returns = []
    disc_ret = 0
    
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        a = epgreedy(epsilon[i], q)
        e = np.zeros(env.n_features)
        done  = False
        cde=0
        while not done:
            features_, r, done = env.step(a)
            q_ = features_.dot(theta)
            a_ = epgreedy(epsilon[i], q)
            pos = np.flatnonzero(q == q.max())
            if a_ in pos:
                ap = a_
            else:
                ap = random.choice(pos)
            e = e + features[a]
            theta = theta + eta[i] * (r + gamma * q_[ap] - q[a]) * e
            if ap == a_: 
                e = gamma_decay[i] * e + features_[ap]
            else:
                e = np.zeros(env.n_features)
            a = a_
            features = features_
            q = q_
            disc_ret += (gamma**(cde - 1))*r
            cde+=1
        returns.append(disc_ret)
    
    if plot:
        return returns
    else:
        return theta 