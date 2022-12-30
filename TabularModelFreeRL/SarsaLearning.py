from _init_.Epsilongreedy import *

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        a = epgreedy(epsilon[i], q[s])
        done = False
        while not done:
            s_, r, done = env.step(a)
            a_ = epgreedy(epsilon[i], q[s_])
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[s_][a_]) - q[s][a])
            s = s_
            a = a_
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value