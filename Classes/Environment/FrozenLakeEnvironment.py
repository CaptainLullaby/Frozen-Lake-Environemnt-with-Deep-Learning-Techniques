from Environment.EnvironmentModel import *
from _init_.contextlib import _printoptions
import numpy as np
       
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        #defining some actions
        self.actions = [0, 1, 2, 3]
        
        #creating a hole count to check if the idicies are holes and add them to a list
        #at the same time adding the boundaries to a new list
        k = 0
        self.hole = []
        #here we initialize four lists that is going to be used to determine the boundary: top, bottom, left and right
        self.boundary_top = []
        self.boundary_left = []
        self.boundary_right = []
        self.boundary_bottom = []

        #code
        for i, row in enumerate(lake):
            for j, char in enumerate(row):
                if(char == "#"):
                    self.hole.append(k)
                
                if(char == "&"):
                    self.start = k
                
                if(char == "$"):
                    self.goal = k
                
                if(i == 0 and j < len(self.lake[0])):
                    self.boundary_top.append(k) #top boundary

                if(k % len(self.lake[0]) == 0):
                    self.boundary_left.append(k) #left boundary
                
                if((k + 1) % len(self.lake[0]) == 0):
                    self.boundary_right.append(k) #right boundary
                
                k += 1
        
        #as there was no way to add the bottom boundary line, i just went with this method
        #in order to find the bottom boundary:
        for i in range(self.boundary_left[-1], self.boundary_right[-1] + 1):
            self.boundary_bottom.append(i)
        
        #corners are special conditions to add two values of the inaccessible moves into the same state
        #we can take the corners of the top and bottom boundaries
        self.corners = [self.boundary_top[0], self.boundary_top[-1], self.boundary_bottom[0], self.boundary_bottom[-1]]
        
        #we can club all the boundaries to a single list, in case if the program requires it
        self.boundary = [self.boundary_top, self.boundary_left, self.boundary_bottom,  self.boundary_right]
        
        #slip_val is used to calculate the probability of the slip in each direction of eah tile
        self.slip_val = self.slip / n_actions

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        

    def p(self, next_state, state, action):
        
        #defining the intial probability
        prob = 0

        #this is to check the positions of the possible movement locatios
        w = state - len(self.lake[0])
        a = state - 1
        s = state + len(self.lake[0])
        d = state + 1
        
        if (state in self.hole or state == self.goal) or state == self.absorbing_state:
            if next_state == self.absorbing_state:
                return 1
            
            else:
                return 0
        
        else:
            #these actions if in the boundary return the added p-val for any move made to go outside the boundary into the same state
            
            #top boundary
            if state in self.boundary_top:
                w = state
                if action == 0 and next_state == state:
                    prob = (1 - self.slip)
            
            else: 
                if action == 0 and next_state == w:
                    prob += (1 - self.slip)
                
            
            
            #left boundary
            if state in self.boundary_left:
                a = state
                if action == 1 and next_state == state:
                    prob += (1 - self.slip)
            
            else: 
                if action == 1 and next_state == a:
                    prob += (1 - self.slip)
            
            
            #bottom boundary
            if state in self.boundary_bottom:
                s = state
                if action == 2 and next_state == state:
                    prob += (1 - self.slip)
            
            else: 
                if action == 2 and next_state == s:
                    prob += (1 - self.slip)
            
            
            #right boundary    
            if state in self.boundary_right:
                d = state
                if action == 3 and next_state == state:
                    prob += (1 - self.slip)
            
            else: 
                if action == 3 and next_state == d:
                    prob += (1 - self.slip)
            
            #probability addition   
            if next_state == w:
                prob += self.slip_val
            
            if next_state == a:
                prob += self.slip_val
            
            if next_state == s:
                prob += self.slip_val
            
            if next_state == d:
                prob += self.slip_val

            return prob

    def r(self, next_state, state, action):
        if(state == self.goal and next_state == self.absorbing_state):
            return 1
        
        else:
            return 0
    

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['↑', '←', '↓', '→']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))