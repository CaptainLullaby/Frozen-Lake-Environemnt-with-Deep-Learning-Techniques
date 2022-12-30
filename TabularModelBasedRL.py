import numpy as np
from Epsilongreedy import *

def policy_evaluation(env, policy, gamma, theta, max_iterations = 0):
    value = np.zeros(env.n_states, dtype=np.float)
    
    states = env.n_states
    
    while True:
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum([env.p(s, ns, policy[s]) * (env.r(s, ns, policy[s]) + gamma * value[ns]) for ns in range(env.n_states)])
            delta = max(delta, np.abs(v - value[s]))
            
            if delta < theta:
                return value
    
def policy_improvement(env, value, gamma):
    
    policy = np.zeros(env.n_states, dtype=int)
    
    flag = True
    for s in range(env.n_states):
        action = env.a(s)
        p = policy[s]
        policy[s] = action[np.argmax([ sum([ env.p(s, ns, a) * (env.r(s, ns, a) + gamma * value[ns]) for ns in range(env.n_states)]) for a in action])]
        
        if p != policy[s]:
            flag = False

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states)
    
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    flag = False
    
    while not flag:
        value = policy_evaluation(env, policy, gamma, theta)
        flag = policy_improvement(env, value, gamma)
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
        
    policy = np.zeros(env.n_states, dtype = int)
    
    
    
    while True:
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum([env.p(s, ns, a) * (env.r(s, ns, a) + gamma * value[ns]) for ns in range(env.n_states)]) for a in range(4)])
            delta = max(delta, abs(v- value[s]))
        
        if delta < theta:
            break
    
    for s in range(env.n_states):
        policy[s] = np.argmax([sum([env.p(s, ns, a) * (env.r(s, ns, a) + gamma * value[s]) for ns in range(env.n_states)]) for a in range(4)])
        
    return policy, value