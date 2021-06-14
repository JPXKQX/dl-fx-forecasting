from dataclasses import dataclass
from typing import Tuple, Union
from datetime import datetime, timedelta
from src.data import constants, utils
from src.features import build_features
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
                      Union[str, datetime]] = None, 
        tick_size: int = 1e4
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
        df = df[~df.index.duplicated(keep='first')]
        
        df['increment'] = df.mid.diff() * tick_size
        df['spread'] = df['spread'] * tick_size
        
        # Skip the first observation of each day
        df['select'] = df['time'].diff() < timedelta(hours=1)
        df = df[df.select].set_index('time').drop('select', axis=1)
        
        if to_invert:
            log.info(f"The currency pair {self.quote.value}/{self.base.value} "
                     "is selected.")
            temp = self.quote
            self.quote = self.base
            self.base = temp
        
        df.attrs = {'base': self.base.value, 
                    'quote': self.quote.value, 
                    'scale': tick_size}
        return df

    def load_dataset(
        self,
        gen_method: str,
        past_ticks: int, 
        ticks_ahead: int,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None, 
        is_pandas: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, tf.data.Dataset]:
        """ Generate a dataset using linspace sampling.

        Args:
            gen_method (str): method for sampling from structured data. Options 
            available are \'linspace\' for linspace sampling of consecutive price 
            bars, and \'event_based\' for sampling using CUSUM filter.
            past_ticks (int): amount of past ticks to consider.
            ticks_ahead (int): forecasting horizon
            period (Tuple): start and end date of the period to consider. 
            Defaults to None, which consider the whole period available.
            is_pandas (bool): Whether to return pandas dataframe or tf.Dataset
            **kwargs: Arguments to pass to generator sampler. For example, the 
            threshold for CUSUM filter.

        Returns:
            tf.data.Dataset: dataset for the specified settings.
        """
        # Get the data
        df = self.read(period)
        
        if is_pandas:
            return self._load_df_dataset(df, gen_method, past_ticks,
                                         ticks_ahead, **kwargs)
        else:
            return self._load_tf_dataset(df, gen_method, past_ticks,
                                         ticks_ahead, **kwargs)
        
    def _load_df_dataset(
        self,
        df: pd.DataFrame,
        gen_method: str,
        past_ticks: int, 
        ticks_ahead: int,
        **kwargs
    ) -> pd.DataFrame:
        if gen_method == 'linspace':
            log.info("Linspace sampling is used.")
            if ('overlapping' in kwargs) and kwargs['overlapping']:
                raise("This option has not been implemented.")
            else:
                df = df.reset_index()
                df['select'] = (df.time.diff() < timedelta(hours=1)).values
                xs, ys = [], []
                while df.shape[0] > past_ticks + ticks_ahead:
                    id_next = df.iloc[1:, :].select.idxmin()
                    if ('overlapping' in kwargs) and kwargs['overlapping']:
                        x, y = build_features.get_xy_overlapping(
                            df.iloc[:id_next, :], past_ticks, ticks_ahead)
                    else:
                        x, y = build_features.get_xy_nonoverlapping(
                            df.iloc[:id_next, :], past_ticks, ticks_ahead)
                    xs.append(x)
                    ys.append(y)
                    df = df.iloc[id_next:, :]
                return pd.concat(xs), pd.concat(ys)

        else:
            raise NotImplementedError(f"The generator method \'{gen_method}\' "
                                      f"is not implemented for returning a "
                                      f"pd.DataFrame.")
    def _load_tf_dataset(
        self,
        data: pd.DataFrame,
        gen_method: str,
        past_ticks: int, 
        ticks_ahead: int,
        **kwargs
    ) -> tf.data.Dataset:
        if gen_method == 'linspace':
            log.info("Linspace sampling is used.")
            generator = self._linspace_generator
            if ('overlapping' in kwargs) and kwargs['overlapping']:
                step = 1
            else:
                step = past_ticks + ticks_ahead  
            args = [data, past_ticks, ticks_ahead, step]
        elif gen_method == 'event_based':
            log.info("Event-based sampling is used.")
            generator = self._event_based_generator
            args = [data, past_ticks, ticks_ahead]
            if 'h' in kwargs.keys(): 
                args.append(kwargs['h'])
            else:
                log.info(f"A dynamic threshold will be used with daily "
                          "volatility estimates.")
        else:
            raise NotImplementedError(f"The generator method \'{gen_method}\' "
                                      f"is not implemented.")
            
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
        ticks_ahead: int,
        step: int = 1
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        i = past_ticks - 1
        total_ticks = past_ticks + ticks_ahead
        while i + ticks_ahead < df.shape[0]:
            x = tf.constant(df[i-past_ticks:i+1, :].ravel('F'))
            y = tf.constant(df[i+total_ticks, -1])
            yield x, y
            i += step


def get_daily_volatility(price: pd.Series, span: int = 100):
    # daily vol, reindexed to close
    df0 = price.index.searchsorted(price.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(price.index[df0 - 1], 
                    index=price.index[price.shape[0]-df0.shape[0]:])
    df0 = price.loc[df0.index] / price.loc[df0.values].values - 1 # daily returns
    df0 = df0.ewm(span=span).std()
    return df0
