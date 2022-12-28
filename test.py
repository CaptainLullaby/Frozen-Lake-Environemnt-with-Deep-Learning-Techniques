from FrozenLakeEnvironment import *
from TabularModelFreeRL import *

#big lake

# lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]


    # Small lake
lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

seed = 0
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
# play(env)

#print('## Sarsa')
max_episodes = 4000
#policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
#env.render(policy, value)

print('')

print('## Q-learning')
policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
env.render(policy, value)