import numpy as np
from FrozenLakeEnvironment import *

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:
    iterations = 0
    delta_1 = np.zeros(1,dtype=np.float)
    action = ['w', 'a', 's', 'd']
    while delta_1 > theta and iterations <= max_iterations:
      delta_1 = np.zeros(1,dtype=np.float)
      iterations += 1
      for i in value:
        v = value[i]
        value[i] = 0
        for a in action:
          next_state = env.step(a)[0]
          value[i] += action[policy[i]]*(env.p(next_state,i,a)*(env.r(next_state,i,a)+gamma*value[next_state]))
        delta_1 = np.max(np.insert(delta_1,0,abs(value[i]-v)))
    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:
    for i in policy:
      delta_2 = np.zeros(1,dtype=np.float)
      action = ['w', 'a', 's', 'd']
      for a in action:
        next_state = env.step(a)[0]
        Q_policy = (env.p(next_state,i,a)*(env.r(next_state,i,a)+gamma*value[next_state]))
        delta_2 = np.insert(delta_2,1,Q_policy)
      max_value_action = action[np.argmax(delta_2)]
      policy[i] = np.argmax(delta_2)
    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    iterations = 0
    delta_3 = np.zeros(env.n_states,dtype=int)
    while delta_3 != 0 and iterations <= max_iterations:
      delta_3 = np.zeros(env.n_states,dtype=int)
      iterations += 1
      value = policy_evaluation(env, policy, gamma, theta, max_iterations)
      new_policy = policy_improvement(env, value, gamma)
      delta_3 = sum(np.abs(np.subtract(policy,new_policy)))
      policy = new_policy
    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    iterations = 0
    action = ['w', 'a', 's', 'd']
    
    while delta_4 > theta and iterations <= max_iterations:
      delta_4 = np.zeros(1,dtype=np.float)
      iterations += 1
      for i in value:
        v = value[i]
        value[i] = []
        for a in action:
          next_state = env.step(a)[0]
          value[i] += (env.p(next_state,i,a)*(env.r(next_state,i,a)+gamma*value[next_state]))
        value[i] = np.max(value[i])
        delta_4 = np.max(np.insert(delta_4,0,abs(value[i]-v)))
    policy = policy_improvement(env, value, gamma)
    return policy, value