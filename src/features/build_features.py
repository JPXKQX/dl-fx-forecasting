from dataclasses import dataclass
from typing import Tuple, List
from datetime import timedelta
from src.data.constants import Currency, ROOT_DIR
from src.data import data_loader

import pandas as pd
import logging

logger = logging.getLogger("Feature Builder")


@dataclass
class FeatureBuilder:
    base: Currency
    quote: Currency
    path: str = f"{ROOT_DIR}/data/"

    def build(
        self,
        freqs: List[int],
        obs_ahead: int,
        period: Tuple[str, str] = None, 
        pair: Tuple[Currency, Currency] = None
    ):
        dl = data_loader.DataLoader(self.base, self.quote, self.path + "raw/")
        df_inc = dl.read(period)[['increment']]
        df = get_features(df_inc.increment, freqs)
        
        # Merge As Of if there is a new currency pair
        if pair is not None:
            dl = data_loader.DataLoader(*pair, self.path + "raw/")
            df_aux = dl.read(period)['increment']
            df_aux = get_features(df_aux, freqs)
            df = pd.merge_asof(df, df_aux, left_index=True, right_index=True)

        first_obs = df.reset_index().time.diff(max(freqs)) < timedelta(hours=1)
        last_obs = df.reset_index().time.diff(-obs_ahead) > timedelta(hours=-1)
        mask = pd.DataFrame((first_obs & last_obs).values, index=df.index)
        X = df[mask[0]]
        y = df_inc.shift(-obs_ahead)[mask[0]]
        return X, y


def get_features(df: pd.DataFrame, freqs_features: List[int]) -> pd.DataFrame:
    features = {}
    for n_obs in freqs_features:
        features[n_obs] = df.ewm(span=n_obs).mean()
    df = pd.DataFrame(features)
    return df
