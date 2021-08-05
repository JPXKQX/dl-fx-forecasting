import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List
from pydantic.dataclasses import dataclass


@dataclass
class StrategySimulator:
    steps: int
    trading_cost_bps: int
    time_cost_bps: int
    """ Implements core trading simulator for single-instrument univ """

    def __post_init_post_parse__(self):
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.pnl = np.zeros(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_mid = np.zeros((2, self.steps))

        # Auxiliary strategies
        self.regr_positions = np.zeros(self.steps)
        self.regr_returns = np.zeros(self.steps)
        self.rf_positions = np.zeros(self.steps)
        self.rf_returns = np.zeros(self.steps)
        self.mlp_positions = np.zeros(self.steps)
        self.mlp_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.pnl.fill(0)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_mid.fill(0)
        self.regr_positions.fill(0)
        self.regr_returns.fill(0)
        self.rf_positions.fill(0)
        self.rf_returns.fill(0)
        self.mlp_positions.fill(0)
        self.mlp_returns.fill(0)

    def take_step(
        self, 
        action: int, 
        mid_prices: np.array
    ) -> Tuple[float, Dict[str, float]]:
        """ Calculates PnLs, trading costs and reward based on an action and latest
        market return and returns the reward and a summary of the day's activity.

        Args:
            action (int): The action to execute. Choices are 0 (short), 1 (not trade) 
                and 2 (long).
            mid_prices (array): Array or tuple with the ask and bid prices. Then, the
            predictions of each model are included.

        Returns:
            float: reward value of at current step
            Dict[str, float]]: additional information
        """
        ask, bid, regr, mlp, rf = mid_prices
        start_position = self.positions[max(0, self.step - 1)]
        regr_start = self.regr_positions[max(0, self.step - 1)]
        mlp_start = self.mlp_positions[max(0, self.step - 1)]
        rf_start = self.rf_positions[max(0, self.step - 1)]
        self.market_mid[:, self.step] = [ask, bid]
        self.actions[self.step] = action

        end_position = action - 1  # -1 (short), 0 (not trade), 1 (long)
        self.positions[self.step] = end_position
        self.trades[self.step] = abs(end_position) # 1 if sell or buy, 0 otherwise

        # Not trading has also a penalty.
        trade_costs = self.trades[self.step] * self.trading_cost_bps
        time_cost = 0 if end_position != 0 else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        if self.step > 0:
            inc = (self.market_mid[0, self.step] - self.market_mid[1, self.step - 1])
            self.pnl[self.step] = start_position * 1e4 * inc              
            self.regr_returns[self.step] = regr_start * 1e4 * inc
            self.rf_returns[self.step] = rf_start * 1e4 * inc
            self.mlp_returns[self.step] = mlp_start * 1e4 * inc
        else:
            self.pnl[self.step] = 0 # It does not matter because start_position is 0.
            self.regr_returns[self.step] = 0
            self.rf_returns[self.step] = 0
            self.mlp_returns[self.step] = 0
        reward = self.pnl[self.step] - self.costs[self.step]
        self.strategy_returns[self.step] = reward

        # Additional
        if regr > bid - ask:
            self.regr_positions[self.step] = 1
        elif regr < ask - bid:
            self.regr_positions[self.step] = -1
        if rf > bid - ask:
            self.rf_positions[self.step] = 1
        elif rf < ask - bid:
            self.rf_positions[self.step] = -1
        if mlp > bid - ask:
            self.mlp_positions[self.step] = 1
        elif mlp < ask - bid:
            self.mlp_positions[self.step] = -1

        info = {
            'reward': reward, 
            'pnl': self.pnl[self.step], 
            'costs': self.costs[self.step]
        }

        self.step += 1
        return reward, info

    def result(self) -> pd.DataFrame:
        """ Get the current result as a daframe.

        Returns:
            pd.DataFrame: current results
        """
        return pd.DataFrame({
            'action': self.actions[:self.step],
            'pnl': self.pnl[:self.step],
            'ask': self.market_mid[0, :self.step],
            'mid': np.mean(self.market_mid[:, :self.step], axis=0),
            'bid': self.market_mid[1, :self.step],
            'strategy_return': self.strategy_returns[:self.step],
            'position': self.positions[:self.step],
            'cost': self.costs[:self.step],
            'trade': self.trades[:self.step],
            'regr_positions': self.regr_positions[:self.step],
            'regr_pnl': self.regr_returns[:self.step],
            'rf_positions': self.rf_positions[:self.step],
            'rf_pnl': self.rf_returns[:self.step],
            'mlp_positions': self.mlp_positions[:self.step],
            'mlp_pnl': self.mlp_returns[:self.step]
        })
