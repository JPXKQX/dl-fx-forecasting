import logging
from src.data.constants import Currency
from pydantic.dataclasses import dataclass
from typing import List, Tuple, Union, Dict

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale
from src.agent.trading_data import TradingDataLoader
from src.agent.trading_strategy import StrategySimulator


logger = logging.getLogger("RL Agent")


@dataclass
class TradingEnv(gym.Env):
    trading_sessions: int = 5000
    trading_cost_bps: float = 0
    time_cost_bps: float = 1e-4
    scaling_difficulty: float = 1.0

    """A simple trading environment for reinforcement learning.
    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG
    Trading has an optional cost (default: 10bps) of the change in position value.
    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __post_init_post_parse__(self):
        self.tdl = TradingDataLoader(
            Currency.EUR, Currency.GBP, 
            ('2020-04-12', '2020-04-18'), 
            200, 5, 
            scaling_difficulty=self.scaling_difficulty,
            aux=(Currency.USD,)
        )
        self.ss = StrategySimulator(
            steps=self.trading_sessions, 
            trading_cost_bps=self.trading_cost_bps,
            time_cost_bps=self.time_cost_bps
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = self.tdl.get_observation_space()
        self.reset()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict]:
        """ Make a step in the action space, and simulte its dynamics. 

        Returns:
            np.array: next observation/state
            float: reward obtained at the current state
            bool: flag indicating the finishing of the episode
            dict: additional information
        """
        assert self.action_space.contains(action), f"{action} {type(action)} invalid"
        observation, done = self.tdl.take_step()
        reward, info = self.ss.take_step(action=action, mid_prices=observation)
        return observation, reward, done, info

    def reset(self):
        """ Resets the status of the environment.

        Returns:
            np.array: first observation
        """
        self.tdl.reset()
        self.ss.reset()
        return self.tdl.take_step()[0]

    def render(self, mode='human'):
        """ Used to display the state when is possible. Not used in this case. """
        pass