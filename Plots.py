from Main._init_._init_ import *
import numpy as np
import matplotlib.pyplot as plt

lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

seed = 0
env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
max_episodes = 4000
linear_env = LinearWrapper(env)
image_env = FrozenLakeImageWrapper(env)

ret = sarsa(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, plot = True, seed=seed)
y = np.convolve(ret, np.ones(20)/20, mode= 'valid')

plt.plot( y, c = "red", label = "Learning curve")
plt.title("Deep Reinforcment Learning")
plt.xlabel(" Number of episodes")
plt.ylabel(" Discounted returns")
plt.legend()
plt.show()

ret = q_learning(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, plot = True, seed=seed)
y = np.convolve(ret, np.ones(20)/20, mode= 'valid')

plt.plot( y, c = "red", label = "Learning curve")
plt.title("Deep Reinforcment Learning")
plt.xlabel(" Number of episodes")
plt.ylabel(" Discounted returns")
plt.legend()
plt.show()

ret = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5,plot = True, seed=seed)
y = np.convolve(ret, np.ones(20)/20, mode= 'valid')

plt.plot( y, c = "red", label = "Learning curve")
plt.title("Deep Reinforcment Learning")
plt.xlabel(" Number of episodes")
plt.ylabel(" Discounted returns")
plt.legend()
plt.show()

ret = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5,plot = True, seed=seed)
y = np.convolve(ret, np.ones(20)/20, mode= 'valid')

plt.plot( y, c = "red", label = "Learning curve")
plt.title("Deep Reinforcment Learning")
plt.xlabel(" Number of episodes")
plt.ylabel(" Discounted returns")
plt.legend()
plt.show()

ret = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001, gamma=gamma,  epsilon=0.2, batch_size=32, target_update_frequency=4, buffer_size=256, kernel_size=3, conv_out_channels=4, fc_out_features=8, seed=4, plot = True)
y = np.convolve(ret, np.ones(20)/20, mode= 'valid')

plt.plot( y, c = "red", label = "Learning curve")
plt.title("Deep Reinforcment Learning")
plt.xlabel(" Number of episodes")
plt.ylabel(" Discounted returns")
plt.legend()
plt.show()