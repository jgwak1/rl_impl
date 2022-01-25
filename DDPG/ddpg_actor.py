# For 'actor policy nework' and 'target actor policy network'

from sympy import Lambda
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda;

import matplotlib.pyplot as plt
import numpy as np


class actor_network:

    def __init__(self) -> None:
        
        self.neural_network = None
        self.actor_optimizer = None


        self.state_dimension = None
        self.action_dimension = None
        self.action_bound = None

        self.model, self.theta = self.states= self.build_network()
        self.target_model, self.target_theta, _ = self.build_network()

        self.dq_da_ph = tf.placeholder( dtype = tf.float32, shape = [None, self.action_dimension], 
                                        name = "dQ/da place-holder which will be computed in critic network." )

        self.dj_dtheta = tf.gradients( ys = self.neural_network.output,   # action outputs from neural network  
                                       xs = self.theta,                                               
                                       grad_ys = -self.dq_da_ph )         # Logic:
                                                                          # Note that "dj_dtheta = da_dtheta * dq_da" 
                                                                          # Since we compute dq_da with critic network
                                                                          # we put a placeholder and start from it as grad_ys,
                                                                          # then compute da_dtheta ( ys = action_output, xs = theta) 
                                                                          # based on it.
        

        '''
        tf.gradients()

        Constructs symbolic derivatives of sum of ys w.r.t. x in xs.

        ys and xs are each a Tensor or a list of tensors.             
        grad_ys is a list of Tensor, holding the gradients received by the ys. The list must be the same length as ys.

        gradients() adds ops to the graph to output the derivatives of ys with respect to xs.
        It returns a list of Tensor of length len(xs) where each tensor is the sum(dy/dx) for y in ys.

        grad_ys is a list of tensors of the same length as ys that holds the initial gradients for each y in ys. 
        When grad_ys is None, we fill in a tensor of '1's of the shape of y for each y in ys. 
        A user can provide their own initial grad_ys to compute the derivatives using a different initial gradient for each y 
        (e.g., if one wanted to weight the gradient differently for each value in each y).

        Args:
        ys: A Tensor or list of tensors to be differentiated.
        xs: A Tensor or list of tensors to be used for differentiation.
        grad_ys: Optional. A Tensor or list of tensors the same size as
            ys and holding the gradients computed for each y in ys.
        ...

        Returns:
        A list of sum(dy/dx) for each x in xs.
        '''


    def build_network(self):

        # build a network with architecture of:
        #
        # input_layer --> hidden_layer-1 (64 Neurons) --ReLU--> hidden_layer-2 (32 Neurons) --ReLU--> hidden_layer-3 (16 Neurons) 
        # --ReLU--> 1 --TanH--> action (1x1)
        #

        state_input = Input( shape= (self.state_dimension, None) )   # https://www.tensorflow.org/api_docs/python/tf/keras/Input
                                                                # shape	A shape tuple (integers), not including the batch size. 
                                                                # For instance, shape=(32,) indicates that the expected input will be
                                                                # batches of 32-dimensional vectors. 
                                                                # Elements of this tuple can be None; 
                                                                # 'None' elements represent dimensions where the shape is not known.

        hidden_layer_1 = Dense( units= 64, activation= 'relu')(state_input)    # https://keras.io/api/layers/core_layers/dense/
                                                                               
                                                                               # units: Positive integer, 
                                                                               #        dimensionality of the output space.
        
        hidden_layer_2 = Dense( units = 32, activation= 'relu')( hidden_layer_1 )
        hidden_layer_3 = Dense( units = 16, activation= 'relu')( hidden_layer_2 )
        output = Dense( units = self.action_dimension, activation = 'tanh')( hidden_layer_3 )
        
        action_output = Lambda(lambda x: x*self.action_bound)(output)            # Lambda() needed for "ValueError: Output tensors to a Model must be the output of a TensorFlow `Layer` (thus holding past layer metadata)."
        neural_network = Model(inputs = state_input, outputs = action_output)    # Model groups layers into an object with training and inference features.
        
        neural_network.summary()
        theta = neural_network.trainable_weights
        
        return neural_network, theta, state_input


    def train_network(self):
        pass

    def predict(self):
        pass
