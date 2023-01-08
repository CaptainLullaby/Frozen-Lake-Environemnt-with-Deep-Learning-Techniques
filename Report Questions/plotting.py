from Main._init_._init_ import *
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

ret = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, plot = True, seed=seed)
y = np.convolve(ret, np.ones(20)/20, mode= 'valid')

plt.plot( y, c = "red", label = "Learning curve")
plt.title("Linear Sarsa Control")
plt.xlabel(" Number of episodes")
plt.ylabel(" Discounted returns")
plt.legend()
plt.show()