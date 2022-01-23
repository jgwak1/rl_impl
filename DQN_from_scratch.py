import tensorflow as tf
import numpy as np
import gym

class DQN:
    def __init__(self, epsilon) -> None:
        # env
        self.env = None

        # architectures 
        self.policy_network = None

        # algorithmic
        self.epsilon = epsilon

    def preprocess(self, obs):
        # preprocess is equivalent to \pi in the pseudo-code.

        # output =  \pi(obs)
        # return output
        pass

    def max_action(self):
        self.policy_network
        pass

    def train(self, episode = 100, time = 100):
        ''' Pseudo-code in https://www.researchgate.net/figure/Pseudo-code-of-DQN-with-experience-replay-method-12_fig11_333197086'''


        # Initialize replay memory D to capacity N
        self.replay_memory = Replay_Buffer()

        # Initialize action-value function Q with random weights
        for eps in (1, episode+1):
            # Initialize squence s_{1} = { x_{1} } and preprocessed sequenced \pi_{1} = \pi ( s_{1} ) 
            # TODO:  
            #      (1) Understand what above means

            obs = self.env.reset()
            for t in (1, time+1):
                
                # Epsilon-Greedy
                if np.random.rand() <= self.epsilon:
                    action = self.max_action(obs)
                else:
                    action = self.env.action_space.sample()
                
                # Execute action a_{t} in emulator and observe reward r_{t} and image x_{t+1}
                # TODO:  
                #      (1) Understand what image x_{t+1} is. 
                obs, reward, done, info = self.env.step(action)
                
                # Set s_{t+1} = s_{t}, a_{t}, x_{t+1} and preprocess \pi_{t+1} = \pi(s_{t+1})
                
                
                if done:
                    break


class Replay_Buffer:
    def __init__(self, capacity = 1024) -> None:
        self.capacity = capacity
        self.count = 0
        self.buffer = [] # list of transition-dicts

    def store(self, transition : dict) -> bool:

        if self.capacity == self.count:
            raise Exception("Unable to add as replay buffer is at its full capacity({})".format(self.capacity))
        if list(transition.keys()) != ['obs','action','reward','next_obs']:
            raise Exception("Transition should contain 'obs','action','reward','next_obs'.")

        self.buffer.append( transition )


