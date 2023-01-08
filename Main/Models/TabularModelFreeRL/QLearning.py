from Main._init_.Epsilongreedy import epgreedy
import numpy as np

def q_learning(env, max_episodes, eta, gamma, epsilon,plot = False, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    returns = []
    disc_ret = 0
    
    for i in range(max_episodes):
        cde = 0
        s = env.reset()
        done = False
        while not done:
            a = epgreedy(epsilon[i], q[s])
            s_, r, done = env.step(a)
            a_ = epgreedy(epsilon[i], q[s_])
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[s_][a_]) - q[s][a])
            s = s_
            disc_ret += (gamma**(env.n_steps - 1))*r
            cde+=1
        returns.append(disc_ret)
     
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    
    if plot:
        return returns
    else:     
        return policy, value