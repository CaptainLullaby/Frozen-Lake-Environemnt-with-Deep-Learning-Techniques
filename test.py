from _init_._init_ import *

#big lake

lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

seed = 0
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
max_episodes = 5000
linear_env = LinearWrapper(env)

print('## Linear Q-learning')

parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
policy, value = linear_env.decode_policy(parameters)
linear_env.render(policy, value)
