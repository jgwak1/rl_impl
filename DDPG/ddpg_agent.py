from ddpg_actor import actor_network
from ddpg_critic import critic_network

from replay_buffer import replay_buffer

import gym
import numpy as np

class ddpg_agent:

    def __init__( self,
                  env : gym.Env, 
                  tau : float = 0.005, epsilon : float = 0.05      
                
                ) -> None:
        
        # env
        self.env = env

        # hyperparams
        self.tau = tau
        self.epislon = epsilon



        self.actor = None
        self.critic = None
        self.target_actor = None 
        self.target_critic = None
        
        self.replay_buffer = None

        pass

    def td_target(self):

    def train(self) -> None:

        '''Source : 수학으로 풀어보는 강화학습 원리와 알고리즘 p.254 (DDPG)'''

        # PSUEDO-CODE

        # Initialize parameters of actor-network and critic-network.
        # Copy the initialized parameters of actor-network and critic-network to the target-actor-network and target-critic-network.
        # Initalize the replay buffer

        # Repeat for epsidoes:
        #
        #  (1) Initalize the action_noise process ( \epsilon ) for exploration
        #  (2) Initialize the state-variable ( x_{t} )for this episode
        #
        #    Repeat for N-timesteps
        #
        #       (1) Compute action based on current actor-policy ( u_{t} = \pi_{\theta}(x_{t}) + \epsilon_{t} )
        #       (2) Apply the computed action to the environment and get the reward and next state.
        #       (3) Store the transition (state, action, reward, next state) to the replay buffer.
        #       (4) Randomly sample N transitions into the minibatch from the replay buffer
        #       (5) Compute the TD target using the reward and target-actor-network and target-critic-network.
        #       (6) Update the critic-network using the loss function which uses the TD target and critic-network.
        #           (6-1) Loss Function = 1/(2*N) * tf.reduce_sum ( ( TD_target - critic_network ( state, action) ) ** 2 ) , 
        #                 given "N" is size of minibatch.
        #       (7) Update the actor-network with the following gradient
        #           (7-1) \gradient_{\theta} J(\theta) \approx tf.reduce_sum( \gradient_{theta} \pi_{\theta} (state) * \gradient_{\u_{i}) Q_{\PHI} )( state, action )})
        #                 given J is objective function, and action = \pi_{\theta}(x_{i})
        #       (8) Update the target-actor-network and target-critic-network
        #           (8-1)  \phi_{t} = \tau * \phi + (1-\tau) * \phi_{t}
        #                  \theta_{t} = \tau * \theta + (1-\tau) * \theta_{t}  

        pass
