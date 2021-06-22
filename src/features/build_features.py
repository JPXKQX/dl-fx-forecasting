from dataclasses import dataclass
from typing import Tuple, List, Union
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
        df_base = dl.read(period)[['increment', 'mid']]
        
        # Merge As Of if there is a new currency pair
        if pair:
            dl = data_loader.DataLoader(*pair, self.path + "raw/")
            df_aux = dl.read(period)['mid']
            df = pd.merge_asof(df_base['mid'], df_aux, left_index=True, 
                               right_index=True)
            df = df.diff().dropna() * 1e4
            df = get_features(df, freqs)
            index0 = df_base.index.get_loc(df.index[0])
            df_base = df_base.iloc[index0:, :]
        else:
            df = get_features(df_base.increment, freqs)

        first_obs = df.reset_index().time.diff(max(freqs)) < timedelta(hours=1)
        last_obs = df.reset_index().time.diff(-obs_ahead) > timedelta(hours=-1)
        mask = pd.DataFrame((first_obs & last_obs).values, index=df.index)
        X = df[mask[0]]
        y = df_base.increment.rolling(obs_ahead).sum().shift(-obs_ahead+1)[mask[0]]
        return X, y


def get_features(
    df: Union[pd.Series,pd.DataFrame], 
    freqs_features: List[int]
) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        features = {}
        for n_obs in freqs_features:
            features[n_obs] = df.ewm(span=n_obs).mean()
        df = pd.DataFrame(features)
    else:
        features1, features2 = {}, {}
        for n_obs in freqs_features:
            features1[n_obs] = df.iloc[:, 0].ewm(span=n_obs).mean()
            features2[n_obs] = df.iloc[:, 1].ewm(span=n_obs).mean()

        df1 = pd.DataFrame(features1)
        df2 = pd.DataFrame(features2)
        df = pd.concat([df1, df2], axis=1)
    return df
