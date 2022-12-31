from _init_._init_ import *

#big lake

lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

seed = 0
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
max_episodes = 4000
#play(env)

print('## Policy iteration')
policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
env.render(policy, value)

print('')

print('## Value iteration')
policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
env.render(policy, value)

print('')

print('# Model-free algorithms')
max_episodes = 4000

print('')

print('## Sarsa')
policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
env.render(policy, value)

print('')

print('## Q-learning')

policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
env.render(policy, value)
print('')


linear_env = LinearWrapper(env)

print('## Linear Sarsa')

parameters = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
policy, value = linear_env.decode_policy(parameters)
linear_env.render(policy, value)