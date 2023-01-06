from _init_._init_ import *
from DeepReinforcedLearning import *

#big lake

lake =  [['&', '.', '.', '.', '.', '.', '.', '.'],
             ['.', '.', '.', '.', '.', '.', '.', '.'],
             ['.', '.', '.', '#', '.', '.', '.', '.'],
             ['.', '.', '.', '.', '.', '#', '.', '.'],
             ['.', '.', '.', '#', '.', '.', '.', '.'],
             ['.', '#', '#', '.', '.', '.', '#', '.'],
             ['.', '#', '.', '.', '#', '.', '#', '.'],
             ['.', '.', '.', '#', '.', '.', '.', '$']]

seed = 0
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
max_episodes = 15000
linear_env = LinearWrapper(env)

#play(env)

#image_env = FrozenLakeImageWrapper(env)

print('')

print('## Policy iteration')
policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
env.render(policy, value)

print('')

print('## Value iteration')
policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
env.render(policy, value)