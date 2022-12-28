from FrozenLakeEnvironment import *

    # Small lake
lake = [['&', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

env = FrozenLake(lake, slip=0.1, max_steps=16, seed=0)
gamma = 0.5
play(env)