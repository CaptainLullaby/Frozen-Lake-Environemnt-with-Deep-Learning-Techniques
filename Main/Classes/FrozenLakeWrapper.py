import numpy as np

class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env

        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]
        
        self.state_image = {env.absorbing_state: np.stack([np.zeros(lake.shape)] + lake_image)}
        
        start_state = np.zeros(lake.shape)
        start_state = np.where(lake == "&", 1, 0)
        goal_state = np.zeros(lake.shape)
        goal_state = np.where(lake == "$", 1, 0)
        hole_state = np.zeros(lake.shape)
        hole_state = np.where(lake == "#", 1, 0)
        
        player_state = np.zeros(lake.shape)
        for state in range(lake.size):
            #player position
            player_state[state % lake.shape[0]][state // lake.shape[1]] = 1
                
            self.state_image[state] = [player_state, start_state, hole_state, goal_state]
            
                

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)