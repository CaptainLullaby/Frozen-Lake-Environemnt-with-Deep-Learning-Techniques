from Main.Classes.ReplayBuffer import *
from Main.Classes.FrozenLakeWrapper import *
from Main.Classes.DeepQNet import *
import numpy as np
        
def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon, batch_size, target_update_frequency, buffer_size, kernel_size, conv_out_channels, fc_out_features, seed, plot = False):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed)

    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    returns = []
    disc_ret = 0
    
    for i in range(max_episodes):
        state = env.reset()
        done = False
        cde = 0
        
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)

            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()

                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)
            
            disc_ret += (gamma**(cde - 1))*reward
            cde+=1

        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())
        
        returns.append(disc_ret)

    if plot:
        return returns
    else:
        return dqn