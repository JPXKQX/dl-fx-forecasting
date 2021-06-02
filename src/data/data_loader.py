from dataclasses import dataclass
from typing import Tuple, Union
from datetime import datetime
from src.data import constants, utils
from src.data.constants import Currency

import dask.dataframe as dd
import pandas as pd
import tensorflow as tf
import os


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
        
        # Read data
        df = dd.read_parquet(folder, filters = filter_dates, 
                             engine="pyarrow-dataset")

        # Preprocess
        df = df.compute()
        df = df.set_index('time')
        if to_invert:
            df[['spread', 'mid']] = df[['spread', 'mid']].rdiv(1)
        
        df.attrs = {'base': self.base.value, 'quote': self.quote.value}
        return df

    def load_dataset(
        self,
        past_ticks: int, 
        ticks_ahead: int,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None
    ) -> tf.data.Dataset:
        """[summary]

        Args:
            past_ticks (int): amount of past ticks to consider.
            ticks_ahead (int): forecasting horizon
            period (Tuple): start and end date of the period to consider. 
            Defaults to None, which consider the whole period available.

        Returns:
            tf.data.Dataset: dataset for the specified settings.
        """
        # Get the data
        df = self.read(period)

        ds = tf.data.Dataset.from_generator(
            self._generator_samples, args=[df, past_ticks, ticks_ahead], 
            output_types=(tf.float32, tf.float32),
            output_shapes=([2 * self.past_ticks, ], []))
        
        return ds

    def _generator_samples(
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
