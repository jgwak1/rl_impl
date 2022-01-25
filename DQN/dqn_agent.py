import tensorflow as tf
import keras

import numpy as np
import gym

from dqn_policy import dqn_policy
from replay_buffer import Replay_Buffer

class dqn_agent:
    def __init__(self, epsilon = 0.03, discount = 0.99, minibatch_size = 64) -> None:
        # env
        self.env = None

        # MLP
        # https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809
        self.policy_network = self.build_network()
        self.optimizer = None

        # algorithmic
        self.epsilon = epsilon
        self.discount = discount
        self.minibatch_size = minibatch_size


    def build_network(self):
        return dqn_policy()

    def preprocess(self, obs):
        # preprocess is equivalent to \pi in the pseudo-code.
        # output =  \pi(obs)
        # return output
        pass

    def max_action(self):
        self.policy_network
        pass

    def q_value(self, obs, action):
        pass

    def gradient_descent_step(self, target : float):
        
        np.gradient(target)
        pass


    def tf_train(self, episode = 100, time = 100) -> None:
    
        mse = tf.reduce_mean( input_tensor= tf.nn.)
        gd_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss = mse)

    def concept_train(self, episode = 100, time = 100) -> None:
        ''' 
        Pseudo-code and explanations in https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c 
        '''
        # Initialize replay memory D to capacity N
        self.replay_memory = Replay_Buffer( minibatch_size = self.minibatch_size )

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
                
                # Store transition ( \pi_{t}, a_{t}, r_{t}, \pi_{t+1} ) in D
                self.replay_memory.store( {"obs":None, "action": action, 'reward': reward, "next_obs": None} )

                # Sample random minibatch of transitions ( \pi_{t}, a_{t}, r_{t}, \pi_{t+1} ) from D
                minibatch = self.replay_memory.random_sample()
                
                if done:
                    target = reward
                    break
                else:
                    target = reward + self.discount * self.q_value( obs = minibatch["next_obs"],
                                                                    action = self.max_action( minibatch["next_obs"] ) )

                # Perform a gradient descent step on ... according to equation 3





