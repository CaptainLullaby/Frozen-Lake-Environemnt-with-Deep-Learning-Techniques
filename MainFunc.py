from _init_._init_ import *

def main():
    
    seed = 0
    
    lake_small = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
    
    lake_large = [['&', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '#', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '#', '#', '.', '.', '.', '#', '.'],
                  ['.', '#', '.', '.', '#', '.', '#', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '$']]
    
    main = True
    lake = lake_small
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    max_episodes = 4000
    gamma = 0.5
    
    while main:
        c = int(input("\n********************\n\tMain Menu:\n********************\n 1. Play!!\n 2. Find optimal using Reinforcement Learning\n 0. Exit :(\n--> "))
        
        if type(c) == int:
            if c == 1:
                flag_1 = True
                while flag_1:
                    opt = int(input("\nChoose:\n 1. Large Lake\n 2. Small Lake\n--> "))
            
                    if type(opt) == int:
                        if opt == 1:
                            lake = lake_large
                            flag_1 = False
                        
                        elif opt == 2:
                            lake = lake_small
                            flag_1 = False
                            
                        else:
                            print("\nPlease choose an available option")
                    else:
                        print("\nAgain an integer please!")
                
                env_1 = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
                play(env_1)
            elif c == 2:
                flag_2 = True
                
                while flag_2:
                    opt_2 = int(input("\n1. Model-Based\n2. Model - Free\n\nDefault chosen lake is small lake(to limit processing time)\n--> "))
                    if type(opt_2) == int:
                        if opt_2 == 1:
                            print("\n## Under-Development ##")
                            pass
                        
                        elif opt_2 == 2:
                            opt_21 = int(input("\n1. Sarsa Contorl\n2. Q-Learning Control\n--> "))
                            if type(opt_21) == int:
                                if opt_21 == 1:
                                    policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
                                    env.render(policy, value)
                                    flag_2 = False
                                
                                elif opt_21 == 2:
                                    policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed)
                                    env.render(policy, value)
                                    flag_2 = False

                            else:
                                print("\nAn integer!!")
                    else:
                        print("\nType an integer!!")
            elif c == 0:
                main = False
            else:
                print("\nThat was not a listed option -_-")
        else:
            print("\nType an integer")






main()