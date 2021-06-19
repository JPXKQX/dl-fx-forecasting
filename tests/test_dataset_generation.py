from src.data.constants import ROOT_DIR, Currency, col_names
from src.data.data_loader import DataLoader
from src.data.data_preprocess import DataPreprocessor, dd
from src.features import get_blocks, build_features
from datetime import datetime
from pytest_mock import MockerFixture
from typing import NoReturn

import dask.dataframe as dd
import tensorflow as tf
import pandas as pd


def mock_parquet():
    df = dd.read_csv(f"{ROOT_DIR}/tests/data/EURUSD-parquet.csv")
    df['time'] = dd.to_datetime(df['time'])
    return df


def mock_raw_csv():
    return dd.read_csv(
        f"{ROOT_DIR}/tests/data/EURUSD-raw.csv", 
        header=None, 
        usecols=[2, 3, 4], 
        names=col_names
    )


class TestDatasetGeneration:
    def prepare(self):
        self.base = Currency.USD
        self.quote = Currency.EUR
        self.path = f"{ROOT_DIR}/data/raw/"
        
    def test_data_loading(self, mocker: MockerFixture) -> NoReturn:
        mocker.patch(
            'src.data.data_loader.dd.read_parquet',
            return_value=mock_parquet()
        )
        mocker.patch.object(
            DataLoader, "_search_pair",
            return_value=("tests/data/EURUSD-parquet.csv", False)
        )
        self.prepare()
        dl = DataLoader(self.base, self.quote, self.path)
        
        # Tests data reading from data storage
        df = dl.read((datetime(2020, 6, 10), datetime(2020, 6, 20)))
        mocker._mocks[0].assert_called_once()
        assert list(df.columns) == ['mid', 'spread', 'increment']
        assert len(df.index) == 99  # The first obs is substracted
        assert df.attrs['base'] == self.base.value
        assert df.attrs['quote'] == self.quote.value
        
        # Test data preparation
        past_ticks = 15
        ds = dl.load_dataset('linspace', past_ticks, 3)
        assert isinstance(ds, tf.data.Dataset)
        assert ds.element_spec[0].shape == tf.TensorShape([past_ticks * 2])
        assert ds.element_spec[1].shape == tf.TensorShape([])
        assert ds.element_spec[0].dtype == tf.float32
        assert ds.element_spec[1].dtype == tf.float32
        
    def test_data_prepocess(self, mocker: MockerFixture) -> NoReturn:
        mocker.patch(
            'src.data.data_preprocess.dd.read_csv',
            return_value=mock_raw_csv()
        )
        mocker.patch.object(
            dd.core.DataFrame, 'to_parquet',
            side_effect=print("Data would have been saved to parquet file.")
        )
        dp = DataPreprocessor(["tests/data/EURUSD-raw.csv"])
        dp._cache_parquet_data()
        mocker._mocks[0].assert_called_once()
        mocker._mocks[1].assert_called_once()


class TestDatasetPreparation:
    def test_instances_overlapping(self):
        past_ticks, ticks_ahead = 10, 1
        df = mock_parquet().compute()
        df['increment'] = df.spread.diff()
        df = df.iloc[1:, :]
        x, y = get_blocks.get_xy_overlapping(df, past_ticks, ticks_ahead)
        assert x.shape == (len(df) - past_ticks - ticks_ahead + 2, 2 * past_ticks)
        assert y.shape[0] == len(df) - past_ticks - ticks_ahead + 2

    def test_instances_nonoverlapping(self):
        past_ticks, ticks_ahead = 10, 1
        df = mock_parquet().compute()
        df['increment'] = df.spread.diff()
        df = df.iloc[1:, :]
        x, y = get_blocks.get_xy_nonoverlapping(df, past_ticks, ticks_ahead)
        assert x.shape == (len(df) // (past_ticks + ticks_ahead), 2 * past_ticks)
        assert y.shape[0] == len(df) // (past_ticks + ticks_ahead)

    def test_features(self, mocker: MockerFixture):
        mocker.patch.object(
            build_features.data_loader.dd, 'read_parquet',
            return_value=mock_parquet())
        mocker.patch.object(
            DataLoader, "_search_pair", return_value=("fake.csv", False)
        )
        freqs = [1, 2, 3, 5]
        fb = build_features.FeatureBuilder(Currency.EUR, Currency.USD)
        X, y = fb.build(freqs, 5, pair=(Currency.EUR, Currency.GBP))
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 2 * len(freqs)
        assert y.shape[1] == 1
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
