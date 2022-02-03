from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model       


class Actor( Model ): # If using neural-network, inherit from "tensorflow.keras.models.Model" class
                      # 2 - By subclassing the Model class: in that case, 
                      # you should define your layers in __init__() and you should implement the model's forward pass in call().
                      # refer to: https://www.tensorflow.org/api_docs/python/tf/keras/Models

    def __init__(self, action_dim: int, action_bound: float) -> None:     
        
        super(Actor, self).__init__()  # init the superclass "tensorflow.keras.models.Model"

        # actor hyperparameter
        self.action_bound = action_bound 
        
        # neural network comps
        # -> you should define your layers in __init__()
        self.h1 = Dense( units= 64, activation= 'relu' )
        self.h2 = Dense( units= 32, activation= 'relu' )
        self.h3 = Dense( units= 16, activation= 'relu' )
        self.action = Dense( units= action_dim, activation= 'tanh')


    def call(self, input_state) -> None: 
        # -> you should implement the model's forward pass in call().
        feed = self.h1( input_state )
        feed = self.h2( feed )
        feed = self.h3( feed )
        action = self.action( feed )

        adjusted_action = Lambda( function = lambda x: x*self.action_bound)
        return adjusted_action



class Critic( Model ):

    def __init__(self) -> None:
        super(Critic, self).__init__()

        # neural network comps
        # -> you should define your layers in __init__()
        self.x1 = Dense(units= 64, activation= 'relu')  # for input_state
        self.x2 = Dense(units= 32, activation= 'relu')  # for input_state
        self.a1 = Dense(units= 32, activation= 'relu')  # for input_action
        
        self.h2 = Dense(units= 32, activation= 'relu')  # this is concatenation layer - should this be 64? 
        self.h3 = Dense(units= 16, activation= 'relu')

        self.q = Dense(1, activation= 'linear')


    def call(self, input_state_action) -> None: 
        # -> you should implement the model's forward pass in call().
        
        input_state = input_state_action[0]
        input_action = input_state_action[1]

        x = self.x1( input_state )
        x = self.x2( x )
        a = self.a1( input_action )

        x_a_concat = concatenate( [x,a], axis= -1 )  # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
        
        feed = self.h2( x_a_concat )
        feed = self.h3( feed )

        q = self.q( feed )

        return q 
                                