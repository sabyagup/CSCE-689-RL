import os
import random
from collections import deque
import tensorflow as tf
from keras import backend as bk
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class DQN(AbstractSolver):
    def __init__(self,env,options):
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.model = self._build_model()
        self.target_model = self._build_model()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        # Add required fields          #
        ################################


    def _build_model(self):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        layers = self.options.layers
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Dense(layers[0], input_dim=state_size, activation='relu'))
        for l in layers:
            model.add(Dense(l, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=self.options.alpha))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            A function that takes a state as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            pass
            

        return policy_fn

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Use:
            self.options.experiment_dir: Directory to save DNN summaries in (optional)
            self.options.replay_memory_size: Size of the replay memory
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps
            self.options.batch_size: Size of batches to sample from the replay memory
            self.env: OpenAI environment.
            self.options.gamma: Gamma discount factor.
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
            new_state, reward, done, _ = self.step(action): To advance one step in the environment
            state_size = self.env.observation_space.shape[0]
            self.model: Q network
            self.target_model: target Q network
            self.update_target_model(): update target network weights = Q network weights
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
                

    def __str__(self):
        return "DQN"

    def plot(self,stats):
        plotting.plot_episode_stats(stats)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            pass
            

        return policy_fn