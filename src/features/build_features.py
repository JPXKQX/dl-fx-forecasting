from dataclasses import dataclass
from itertools import product
from typing import Tuple, List, Union
from datetime import timedelta

from numpy.core.fromnumeric import var
from src.data.constants import Currency, ROOT_DIR
from src.data import data_loader

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Feature Builder")


@dataclass
class FeatureBuilder:
    base: Currency
    quote: Currency
    path: str = f"{ROOT_DIR}/data/"

    def get_aux_currency_pairs(self, aux):
        pairs = []
        if aux:
            if isinstance(aux, Currency):
                pairs.extend(self.get_pairs_to_compute_implicit(aux))
            elif isinstance(aux, (list, tuple)):
                if len(aux) == 1:
                    pairs.extend(self.get_pairs_to_compute_implicit(aux))
                elif len(aux) == 2:
                    pairs.append(aux)
                else:
                    raise ValueError(f'{", ".join(aux)} has been passed as '
                                     f'auxiliary.')
        return pairs

    def get_pairs_to_compute_implicit(
        self, 
        aux: Currency
    ) -> List[Tuple[Currency, Currency]]:
        if isinstance(aux, (list, tuple)): aux = aux[0]
        if aux in [self.base, self.quote]:
            raise ValueError("If selecting one auxiliary currency it must be "
                             "different from the base and quote ones because "
                             "it is used to compute the implicit mid price.")
        return [(self.base, aux), (aux, self.quote)]

    def load_synchronized(self, aux, period):
        pairs = self.get_aux_currency_pairs(aux)
        
        dl = data_loader.DataLoader(self.base, self.quote, self.path + "raw/")
        df = dl.read(period)
        increments = df[['increment']]
        
        # Swap order if base/quote order has been inverted during loading
        if df.attrs['base'] != self.base.value:
            pairs = reversed(pairs)

        for pair in pairs:
            logger.debug(f"Loading dataframe for currency pair "
                         f"{pair[0].value + pair[1].value}.")
            # Read auxiliary currency pair
            dl = data_loader.DataLoader(*pair, self.path + "raw/")
            aux_df = dl.read(period)

            # Merge new currency with loaded data.
            attrs = df.attrs
            aux_str = aux_df.attrs['pair']
            df = pd.merge_asof(df, aux_df, left_index=True, right_index=True,
                               suffixes=("", f"_{aux_str}"))
            if 'pairs' in attrs:
                attrs['pairs'].append(aux_str)
            else:
                attrs['pairs'] = [aux_str]
            df.attrs = attrs
        
        return df.dropna(), increments

    def compute_implicit_midprice(self, df):
        if 'pairs' not in df.attrs:
            logger.info('Computation of implicit mid prices not available when '
                        'no auxiliary currency is selected.')
            return df
        elif len(df.attrs['pairs']) == 1:
            logger.info('Computation of implicit mid prices not available when '
                        'one currency pair is selected.')
            return df
        
        # Computing implicit mid prices, dropping aux mid prices and increments
        pairs = df.attrs['pairs']
        base, quote = df.attrs['base'], df.attrs['quote']
        implicit_base = df[f'mid_{pairs[0]}']
        implicit_quote = df[f'mid_{pairs[1]}']
        
        if base == pairs[0][:3]:
            logger.debug("Currency pair between base and auxiliary currencies "
                         "is fine.")
        elif base == pairs[0][3:]:
            logger.debug("Currency pair between base and auxiliary currencies "
                         "is swapped.")
            implicit_base = implicit_base.rdiv(1)
        else:
            logger.error(f"{base} should be in {pairs[0]}")
            raise ValueError(f"{base} should be in {pairs[0]}")

        if quote == pairs[1][:3]:
            logger.debug("Currency pair between auxiliary and quote currencies "
                         "is swapped.")
            implicit_quote = implicit_quote.rdiv(1)
        elif quote == pairs[1][3:]:
            logger.debug("Currency pair between auxiliary and quote currencies "
                         "is fine.")
        else:
            logger.error(f"{quote} should be in {pairs[1]}")
            raise ValueError(f"{quote} should be in {pairs[1]}")

        df['implicit_mid'] = implicit_base * implicit_quote
        df['implicit_increment'] = df['implicit_mid'].diff()
        
        # Drop intermediary data variables
        return df.drop(list(map(
            lambda x: "_".join(x), 
            product(["mid", "increment"], df.attrs['pairs']))), axis=1).dropna()


    def build(
        self,
        freqs: List[int],
        obs_ahead: int,
        period: Tuple[str, str] = None, 
        aux_currencies: Tuple[Currency, ...] = None, 
        variables: List[str] = None,
        quantile: float = None,
    ):
        df, incs = self.load_synchronized(aux_currencies, period)
        df = self.compute_implicit_midprice(df)
        
        # Select columns
        if variables:
            is_var_chosen = lambda x: any([(v in x) for v in variables])
            mask = list(filter(is_var_chosen, df.columns.values))
            df = df.loc[:, mask]

        # Filter by dates from 7:00 AM to 7:00 PM
        df['filter'] = df.index.where(df.index.hour.isin(list(range(7, 19))))
        df = df.dropna().drop('filter', axis=1)

        if isinstance(freqs, list):
            df = get_features(df, freqs)
        else:
            df = get_x_blocks(df, freqs)
        
        # Filtering by spread values. Drop values over a quantile specified
        if quantile:
            quantiles = df['spread'].quantile(quantile)
            spreads = df['spread'].where(df['spread'] <= quantiles).dropna()
            indices = spreads.index
            df = df.loc[indices]

        num_prev = max(freqs) if isinstance(freqs, list) else freqs
        first_obs = df.reset_index().time.diff(num_prev) < timedelta(hours=2)
        last_obs = df.reset_index().time.diff(-obs_ahead) > timedelta(hours=-2)
        mask = pd.DataFrame((first_obs & last_obs).values, index=df.index)
        indices = mask.where(mask).dropna().index 
        X = df[mask[0]]
        fut_inc = incs[['increment']].rolling(obs_ahead).sum().shift(-obs_ahead)
        y = fut_inc.loc[indices]
        return X, y


def get_features(
    df: Union[pd.Series,pd.DataFrame], 
    freqs_features: List[int]
) -> pd.DataFrame:
    n_vars = len(df.columns)
    if isinstance(df, pd.Series):
        features = {}
        for n_obs in freqs_features:
            features[n_obs] = df.ewm(span=n_obs).mean()
        df = pd.DataFrame(features)
    else:
        n_variables = len(df.columns)
        features = []
        for n_obs in freqs_features:
            features.append(df.ewm(span=n_obs).mean())

        df = pd.concat(features, axis=1)

        # Reorder columns
        n_features = len(freqs_features)
        indices = np.arange(n_variables * n_features)
        indices = indices.reshape((n_features, n_variables)).transpose().ravel()
        df = df.iloc[:, list(indices)]

    columns = list(map(lambda x: f"{x[0]}_{str(x[1])}", 
                       zip(df.columns.values, freqs_features * n_vars)))
    df.columns = columns
    return df


def get_x_blocks(df: pd.DataFrame, past_obs: int) -> pd.DataFrame:
    n_variables = len(df.columns)
    data = df.iloc[:-(df.shape[0] % past_obs)].values
    data = data.reshape((-1, past_obs, n_variables)).transpose((0, 2, 1))
    data = np.flip(data, 2).reshape((-1, past_obs * n_variables))

    # Get indices
    indices = df.iloc[list(range(past_obs - 1, df.shape[0], past_obs))].index
    columns = list(map(lambda x: f"{x[0]}_{str(x[1])}", 
                       product(df.columns.values, list(range(1, past_obs+1)))))

    return pd.DataFrame(data, columns=columns, index=indices)


if __name__ == '__main__':
    FeatureBuilder(Currency.GBP, Currency.EUR).run([1, 2, 3, 50], 5, ('2020-04-01', '2020-04-10'), (Currency.USD))
