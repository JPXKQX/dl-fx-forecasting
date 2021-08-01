import logging
import pickle
import pandas as pd
import numpy as np

from pydantic.dataclasses import dataclass
from src.features.build_features import FeatureBuilder
from src.models.neural_network import MultiLayerPerceptron
from src.data.constants import Currency, ROOT_DIR
from gym import spaces
from typing import List, Tuple, Union


logger = logging.getLogger("RL Agent")


@dataclass
class TradingDataLoader:
    """
    Data source for TradingEnvironment

    Loads the inputs for the models considered and prepare the outputs of this models
    as the observations of the state space. It also prepares the bid and ask prices for 
    each timestamp.

    Attrs:
        base (Currency): The base currency.
        quote (Currency): The quote currency.
        period (Tuple[str, str]): The period to extract the observations.
        freqs (Union[List[int], int]): The frequencies to extract.
        horizon (int): The horizon of the features to predict. It is also considered for
        skipping the observation and avoid overlapping rewards.
        variables (List[str]): Set of vairables to extract for model inputs.
        scaling_difficulty (float): The scaling difficulty of the problem. Defaults to 1
        , which means that real word trading at bid and ask. O represent frictionless
        market where all trades are consided at mid price.
        aux (Tuple[Currency, ]): Auxiliary currencies to consider for model inputs.
    """
    base: Currency
    quote: Currency
    period: Tuple[str, str]
    freqs: Union[List[int], int]
    horizon: int
    scaling_difficulty: float = 1.0
    trading_sessions: int = 5000
    aux: Tuple[Currency, ...] = None

    def __post_init_post_parse__(self):
        self.regr, self.mlp, self.rf = None, None, None
        self.load_models()
        self.data = self.load_data()
        self.step, self.offset = 0, None

    def load_models(self):
        regr_path = f"{ROOT_DIR}/models/increment/ElasticNet/EURGBP/USD" \
                    f"/increment_difference/" \
                    f"ElasticNet_EURGBP_200-{self.horizon}_20200405-20200411.pkl"
        with open(regr_path, 'rb') as mo_path:
            self.regr = pickle.load(mo_path)

        rf_path = f"{ROOT_DIR}/models/increment/RandomForest/EURGBP/USD" \
                  f"/increment_difference_spread/" \
                  f"RandomForest_EURGBP_200-{self.horizon}_20200405-20200411.pkl"
        with open(rf_path, 'rb') as mo_path:
            self.rf = pickle.load(mo_path)

        mlp_path = f"{ROOT_DIR}/models/increment/MultiLayerPerceptron/EURGBP/USD" \
                   f"/increment_difference/" \
                   f"MultiLayerPerceptron_EURGBP_200-{self.horizon}_20200405-20200411.h5"
        self.mlp = MultiLayerPerceptron()
        self.mlp.load(mlp_path)      

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data for {self.base}/{self.quote}...")
        fb = FeatureBuilder(self.base, self.quote)
        X, y = fb.build(
            [1, 2, 3, 5, 10, 25, 50, 100, 200], self.horizon, 'increment', self.period,
            self.aux, ['increment', 'difference', 'spread', 'mid'])

        mask1 = [(
                ('increment' in x) or ('difference' in x)
            ) and (
                'implicit_increment' not in x
            ) for x in X.columns.values
        ]
        mask2 = [
            ('increment' in x) or ('difference' in x) or ('spread' in x) 
            for x in X.columns.values
        ]

        data = pd.DataFrame(index=X.index)
        data['ask'] = X['mid_1'] - self.scaling_difficulty * (1e-4 * X['spread_1'] / 2)
        data['bid'] = X['mid_1'] + self.scaling_difficulty * (1e-4 * X['spread_1'] / 2)

        # Process ElasticNet - with USD aux - 
        # Features: increment & differnce $ !implicit-increment as input
        # EWMA 1, 2, 3, 5, 10, 25, 50, 100, 200
        data['Regr_5'] = self.regr.predict(X.loc[:, mask1])

        # Process MultiLayerPerceptron - with USD aux - 
        # Features: increment & differnce $ !implicit-increment as input
        # EWMA 1, 2, 3, 5, 10, 25, 50, 100, 200
        data['Mlp_5'] = self.mlp.predict(X.loc[:, mask1])

        # Process RandomForest - with USD aux - increment & difference & spread - 
        # with features EWMA 1, 2, 3, 5, 10, 25, 50, 100, 200
        data['Rf_5'] = self.rf.predict(X.loc[:, mask2])

        # Include spread prediction from InceptionTime

        logger.info(f"Data processed for {self.base}/{self.quote}.")        
        return data

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - 5 * self.trading_sessions
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0
        logger.debug(f"Reseting DataSource, and setting "
                     f"{self.data.index[self.offset]} as starting point.")

    def take_step(self) -> Tuple[np.array, bool]:
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 5  # Avoid overlapping rewards
        done = self.step > self.trading_sessions
        return obs, done

    def get_observation_space(self):
        return spaces.Box(self.data.min(), self.data.max())


if __name__ == '__main__':
    period = ('2020-04-12', '2020-04-18')
    varnames = ['increment', 'difference', 'spread']
    ds = TradingDataLoader(
        Currency.EUR, Currency.GBP, period, 200, 5, varnames, aux=(Currency.USD,)
    )
    ds.reset()
    print(ds.take_step())
