import numpy as np
import tensorflow as tf

from random import sample
from pydantic.dataclasses import dataclass
from collections import deque
from typing import Tuple
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


@dataclass
class DDQN:
    """ DDQN Agent which uses a MLP model to estimate the Q-value.
    
    Attrs:
        state_dim (int): The number of dimensions in the state space.
        num_actions (int): The number of possible actions.
        learning_rate (float): The learning rate of the Q-network
        gamma (float): The discount factor over time of the rewards.
        epsilon_start (float): Initial epsilon of e-greedy policy
        epsilon_end (float): Final epsilon of e-greedy policy
        epsilon_decay_steps (int): The number of decay steps of the e-greedy policy.
        epsilon_exponential_decay (float):
        architecture (Tuple[int, ]): Architecture of the Q-network, where each element
            of the tuple represents the number of neurons in one dense layer.
        l2_reg (float): The l2 regularization term.
        dropout_rate (float): Dropout rate of the layers in the Q-network.
        tau (int): The update frequency of the target network.
        batch_size (int): The batch size of the data.
    """
    state_dim: int
    num_actions: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_end: float
    epsilon_decay_steps: int
    epsilon_exponential_decay: float
    replay_capacity: int
    architecture: Tuple[int, ...]
    l2_reg: float
    dropout_rate: float
    tau: int
    batch_size: int

    def __post_init_post_parse__(self):
        self.experience = deque([], maxlen=self.replay_capacity)

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()


        self.epsilon_decay = (self.epsilon - self.epsilon_end) / \
            self.epsilon_decay_steps
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.losses = []
        self.idx = tf.range(self.batch_size)
        self.train = True

    def build_model(self, trainable: bool = True) -> Sequential:
        """ Constructs the MLP model or Q-network to train the Q-value estimation.

        Args:
            trainable (bool, optional): Flag of whether parameters are updated or not. 
                Defaults to True.

        Returns:
            tf.keras.Model: the Q-network
        """
        layers = []
        for i, units in enumerate(self.architecture):
            input_dim = None if i > 0 else self.state_dim
            layers.append(
                Dense(units, input_dim=input_dim, activation='relu',
                      kernel_regularizer=l2(self.l2_reg), trainable=trainable)
            )
        layers.append(Dropout(self.dropout_rate))
        layers.append(
            Dense(units=self.num_actions, trainable=trainable, name='Q-estimate')
        )
        model = Sequential(layers, name='Q-network')
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(
            next_q_values_target,
            tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1)
        )

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[[self.idx, actions]] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()