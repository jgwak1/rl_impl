import numpy as np
import matplotlib.pyplot as plt
import gym

from ddpg_AC import Actor, Critic
from replay_buffer import replay_buffer



class ddpg_Agent(object):

    def __init__( self, 
                  env : gym.Env, 
                  gamma = 0.95, batch_size = 32, buffer_size = 20000, actor_lr = 0.0001, critic_lr = 0.001, tau = 0.001
                ) -> None:
        
        # hyperparameters
        self.GAMMA = gamma
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.ACTOR_LEARNING_RATE = actor_lr
        self.CRITIC_LEARNING_RATE = critic_lr
        self.TAU = tau

        # env
        self.env = env

        # state and action dimensions and action bound
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = env.action_space.high[0] # highest possible action value

        # actor NN, target-actor NN, critic NN, target-critic NN  

        # -> instantiate the class object which wraps each of 4 neural-networks
        self.actor = Actor( action_dim = self.action_dim, action_bound= self.action_bound)
        self.target_actor = Actor( action_dim = self.action_dim, action_bound= self.action_bound)
        self.critic = Critic()
        self.target_critic = Critic()

        # build the actual model (neural network)
        self.actor.build( input_shape= (None, self.state_dim) ) # Builds the model based on input shapes received.
                                                                # This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.
                                                                # This method only exists for users who want to call model.build() in a stand-alone way 
                                                                # (as a substitute for calling the model on real data to build it). 
                                                                # It will never be called by the framework 
                                                                # (and thus it will never throw unexpected errors in an unrelated workflow).

                                                                # Args:
                                                                # input_shape: Single tuple, TensorShape, or list/dict of shapes, where shapes are tuples, integers, or TensorShapes.
        self.target_actor.build( input_shape= (None, self.state_dim) )

        





    '''
    def td_target(self):



        pass

    def train(self) -> None:


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
'''