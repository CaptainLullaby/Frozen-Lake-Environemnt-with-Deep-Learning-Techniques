import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations = 0):
    value = np.zeros(env.n_states, dtype=np.float)
    
    p = env.p
    r = env.r
    states = env.n_states
    
    while True:
        delta = 0
        for s in env.n_states:
            v = value[s]
            value[s] = sum([p(s, ns, policy[s]) * (r(s, ns, policy[s]) + gamma * value[ns]) for ns in env.n_states])
            delta = max(delta, np.abs(v - value[s]))
            
            if delta < theta:
                return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    p = env.p
    r = env.r
    action = env.a  ##########? what is this
    
    flag = True
    for s in env.n_states:
        act = action(s)
        p = policy(s)
        policy[s] = act[np.argmax([ sum([p(s, ns, a) * (r(s, ns, a) + gamma * value[ns]) for ns in env.n_states]) for a in act])]
        
        if p != policy[s]:
            flag = False

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states)
    
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    action = env.a  ##########? what is this
    
    flag = False
    
    while not flag:
        value = policy_evaluation(env, policy, gamma, theta)
        stable = policy_improvement(env, value, gamma)
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    
    p = env.p
    r = env.r
    act = env.a
    
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    while True:
        delta = 0
        for s in env.n_states:
            actions = act(s)
            v = value[s]
            value[s] = max([sum([p(s, ns, a) * (r(s, ns, a) + gamma * value[ns]) for ns in env.n_states]) for a in actions])
            delta = max(delta, abs(v- value[s]))
        
        if delta < theta:
            break
        
    policy = np.zeros(len(env.n_states), dtype = int)
    
    for s in env.n_states:
        actions = act(s)
        policy[s] = actions[np.argmax([sum([p(s, ns, a) * (r(s, ns, a) + gamma * value[s]) for ns in env.n_states]) for a in actions])]
        
    return policy, value