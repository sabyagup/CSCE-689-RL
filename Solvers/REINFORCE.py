import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Softmax, Input
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from skimage.transform import resize
from skimage import color

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting

   
def pg_loss(rewards):
    def loss(labels, predicted_output):
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        pass
        
        
    return loss

keras.losses.pg_loss = pg_loss

class Reinforce(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = (4,)
        self.trajectory = []
        self.model = self.build_model()
        self.policy = self.create_greedy_policy()

    def build_model(self):
        rewards = Input(shape=(1,))
        
        states = Input(shape=self.state_size)
        d1 = Dense(64, activation='relu')(states)
        d2 = Dense(64, activation='relu')(d1)
        d3 = Dense(64, activation='relu')(d2)
        do = Dense(self.env.action_space.n)(d3)
        out = Softmax()(do)
        
        opt = Adam(lr=self.options.alpha)
        model = Model(inputs=[states, rewards], outputs=out)
        model.compile(optimizer=opt, loss=pg_loss(rewards))
        return model

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            return self.model.predict([[state], np.zeros((1, 1))])[0]

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the REIFORCE algorithm

        Use:
            self.step(action): Performs an action in the env.
            self.env.reset(): Resets the env. 
            self.options.gamma: Gamma discount factor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        pass
        

    def __str__(self):
        return "REINFORCE"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
