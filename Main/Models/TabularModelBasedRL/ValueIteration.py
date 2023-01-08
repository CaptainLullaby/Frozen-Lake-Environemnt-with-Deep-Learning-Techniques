import numpy as np

def value_iteration(env, gamma, theta, max_iterations, value=None):
    
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float32)
        
    policy = np.zeros(env.n_states, dtype = int)
    iter = 0
    
    
    while iter < max_iterations:
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum([env.p(ns, s, a) * (env.r(ns, s, a) + gamma * value[ns]) for ns in range(env.n_states)]) for a in range(env.n_actions)])
            delta = max(delta, abs(v- value[s]))
        
        if delta < theta:
            break
        iter += 1
    
    print("\nNo of iters = ", iter)
    
    for s in range(env.n_states):
        policy[s] = np.argmax([sum([env.p(ns, s, a) * (env.r(ns, s, a) + gamma * value[ns]) for ns in range(env.n_states)]) for a in range(env.n_actions)])
        
    return policy, value