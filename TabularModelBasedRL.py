import numpy as np
from Epsilongreedy import *

def policy_evaluation(env, policy, gamma, theta, max_iterations = 0):
    value = np.zeros(env.n_states, dtype=np.float)
    
    iter = 0
    
    while iter < max_iterations:
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum([env.p(ns, s, policy[s]) * (env.r(ns, s, policy[s]) + gamma * value[ns]) for ns in range(env.n_states)])
            delta = max(delta, np.abs(v - value[s]))
            
        if delta < theta:
            return value
        
        iter +=1
    
def policy_improvement(env, policy, value, gamma):
    
    flag = True
    for s in range(env.n_states):
        action = env.a(s)
        p = policy[s]
        policy[s] = action[np.argmax([ sum([ env.p(ns, s, a) * (env.r(ns, s, a) + gamma * value[ns]) for ns in range(env.n_states)]) for a in range(env.n_actions)])]
        if p != policy[s]:
            flag = False

    return flag
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states)
    
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    flag = False
    
    while not flag:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        flag = policy_improvement(env, policy, value, gamma)
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
        
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
    
    for s in range(env.n_states):
        policy[s] = np.argmax([sum([env.p(ns, s, a) * (env.r(ns, s, a) + gamma * value[ns]) for ns in range(env.n_states)]) for a in range(env.n_actions)])
        
    return policy, value