from _init_._init_ import *
from DeepReinforcedLearning import *

#big lake

lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

seed = 0
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
max_episodes = 4000
linear_env = LinearWrapper(env)

#play(env)

image_env = FrozenLakeImageWrapper(env)


print('## Deep Q-network learning')

dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001, gamma=gamma,  epsilon=0.2, batch_size=32, target_update_frequency=4, buffer_size=256, kernel_size=3, conv_out_channels=4, fc_out_features=8, seed=4)
policy, value = image_env.decode_policy(dqn)
image_env.render(policy, value)
