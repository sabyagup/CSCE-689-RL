import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Softmax, Input
from keras.optimizers import Adam
from keras.losses import huber_loss
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from skimage.transform import resize
from skimage import color

from collections import deque

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


def actor_loss(deltas):
    def loss(labels, predicted_output):
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        pass
        

    return loss
        
keras.losses.actor_loss = actor_loss

class A2C(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = (4,)
        self.trajectory = []
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.policy = self.create_greedy_policy()

    def build_actor(self):
        deltas = Input(shape=(1,))
        
        states = Input(shape=self.state_size)
        d1 = Dense(64, activation='relu')(states)
        d2 = Dense(64, activation='relu')(d1)
        d3 = Dense(64, activation='relu')(d2)
        do = Dense(self.env.action_space.n)(d3)
        probs = Softmax()(do)

        actor = Model(inputs=[states, deltas], outputs=probs)
        actor.compile(optimizer=Adam(lr=self.options.alpha), loss=actor_loss(deltas))

        return actor

    def build_critic(self):
        states = Input(shape=self.state_size)
        d1 = Dense(64, activation='relu')(states)
        d2 = Dense(64, activation='relu')(d1)
        d3 = Dense(64, activation='relu')(d2)
        values = Dense(1, activation='linear')(d3)

        critic = Model(inputs=[states], outputs=values)
        critic.compile(optimizer=Adam(lr=self.options.alpha), loss=huber_loss)

        return critic

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            return self.actor.predict([[state], np.zeros((1, 1))])[0]

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm

        Use:
            self.step(action): Performs an action in the env.
            self.env.reset(): Resets the env. 
            self.options.gamma: Gamma discount factor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   
        ################################
        

    def __str__(self):
        return "A2C"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
