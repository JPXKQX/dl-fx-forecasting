from dataclasses import dataclass
from typing import Tuple, Union
from tqdm import tqdm
from datetime import datetime
from src.data import constants, utils
from src.data.constants import Currency

import dask.dataframe as dd
import pandas as pd
import tensorflow as tf
import os
import logging


log = logging.getLogger("DataLoader")


@dataclass
class DataLoader:
    base: Currency
    quote: Currency
    dir: str = f"{constants.ROOT_DIR}/data/raw/"
    
    def _search_pair(self) -> Tuple[str, bool]:
        if os.path.isdir(self.dir + self.base.value + self.quote.value):
            return self.dir + self.base.value + self.quote.value, False
        elif os.path.isdir(self.dir + self.quote.value + self.base.value):
            return self.dir + self.quote.value + self.base.value, True
        else:
            raise ValueError(f"Could not find data for "
                             f"{self.base.value}/"
                             f"{self.quote.value}")
    
    def read(
        self, 
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """ Read the data for the currency pair for a period of time.
        
        Args:
            period (Tuple): start and end date of the period.
            
        Returns:
            pd.DataFrame: the data for the currency pair
        """
        folder, to_invert = self._search_pair()
        filter_dates = None
        if period is not None:
            start = utils.str2datetime(period[0])
            end = utils.str2datetime(period[1])
            filter_dates = [('time', '>=', start), ('time', '<=', end)]
            log.info(f"Data between {start} and {end} is considered.")
        
        # Read data
        df = dd.read_parquet(folder, filters = filter_dates, 
                             engine="pyarrow-dataset")

        # Preprocess
        df = df.compute()
        df = df.set_index('time')
        df = df[~df.index.duplicated(keep='first')]
        if to_invert:
            df[['spread', 'mid']] = df[['spread', 'mid']].rdiv(1)
        
        df.attrs = {'base': self.base.value, 'quote': self.quote.value}
        return df

    def load_dataset(
        self,
        gen_method: str,
        past_ticks: int, 
        ticks_ahead: int,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None, 
        **kwargs
    ) -> tf.data.Dataset:
        """ Generate a dataset using linspace sampling.

        Args:
            gen_method (str): method for sampling from structured data. Options 
            available are \'linspace\' for linspace sampling of consecutive price 
            bars, and \'event_based\' for sampling using CUSUM filter.
            past_ticks (int): amount of past ticks to consider.
            ticks_ahead (int): forecasting horizon
            period (Tuple): start and end date of the period to consider. 
            Defaults to None, which consider the whole period available.
            **kwargs: Arguments to pass to generator sampler. For example, the 
            threshold for CUSUM filter.

        Returns:
            tf.data.Dataset: dataset for the specified settings.
        """
        # Get the data
        df = self.read(period)

        if gen_method == 'linspace':
            log.info("Linspace sampling is used.")
            generator = self._linspace_generator
            args = [df, past_ticks, ticks_ahead]
        elif gen_method == 'event_based':
            log.info("Event-based sampling is used.")
            generator = self._event_based_generator
            args = [df, past_ticks, ticks_ahead]
            if 'h' in kwargs.keys(): 
                args.append(kwargs['h'])
            else:
                log.info(f"A dynamic threshold will be used with daily "
                             "volatility estimates.")
        else:
            raise NotImplementedError(f"The generator method \'{gen_method}\' "
                                      "is not implemented.")
            
        ds = tf.data.Dataset.from_generator(
            generator, args=args, 
            output_types=(tf.float32, tf.float32),
            output_shapes=([2 * past_ticks, ], []))
        
        return ds
    
    def _event_based_generator(
        self, 
        df: pd.DataFrame, 
        past_ticks: int, 
        ticks_ahead: int,
        h: float = 0.001
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        cusum_pos, cusum_neg = 0, 0
        diff = df.mid.diff()
        for i, ind in enumerate(diff.index[past_ticks:-ticks_ahead], 
                                start=past_ticks):
            cusum_pos = max(0, cusum_pos + diff.loc[ind])
            cusum_neg = min(0, cusum_neg + diff.loc[ind])
            if cusum_neg < -h:
                cusum_neg = 0
                x = tf.constant(df.iloc[i-past_ticks+1:i+1, :].ravel('F'))
                y = tf.constant(df.iloc[i+ticks_ahead, -1])
                yield x, y
            elif cusum_pos > h:
                cusum_pos = 0
                x = tf.constant(df.iloc[i-past_ticks+1:i+1, :].ravel('F'))
                y = tf.constant(df.iloc[i+ticks_ahead, -1])
                yield x, y

    def _linspace_generator(
        self, 
        df: pd.DataFrame, 
        past_ticks: int, 
        ticks_ahead: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        total_ticks = past_ticks + ticks_ahead
        n_samples = df.shape[0] - total_ticks + 1  # with overlapping instances
        for i in range(n_samples):
            x = tf.constant(df[i:i+past_ticks, :].ravel('F'))
            y = tf.constant(df[i+total_ticks-1, -1])
            yield x, y


def get_daily_volatility(price: pd.Series, span: int = 100):
    # daily vol, reindexed to close
    df0 = price.index.searchsorted(price.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(price.index[df0 - 1], 
                    index=price.index[price.shape[0]-df0.shape[0]:])
    df0 = price.loc[df0.index] / price.loc[df0.values].values - 1 # daily returns
    df0 = df0.ewm(span=span).std()
    return df0


dl = DataLoader(Currency.EUR, Currency.USD)
dl.load_dataset(100, 10, ('2020-04-01', '2020-06-01'))
