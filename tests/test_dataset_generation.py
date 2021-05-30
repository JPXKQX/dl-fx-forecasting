from src.data.constants import ROOT_DIR, Currency, col_names
from src.data.data_loader import DataLoader
from src.data.data_preprocess import DataPreprocessor, dd
from datetime import datetime
from pytest_mock import MockerFixture
from typing import NoReturn

import dask


def mock_parquet():
    df = dask.dataframe.read_csv(f"{ROOT_DIR}/tests/data/EURUSD-parquet.csv")
    return df


def mock_raw_csv():
    return dask.dataframe.read_csv(
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
        df = dl.read((datetime(2020, 6, 10), datetime(2020, 6, 20)))
        mocker._mocks[0].assert_called_once()
        assert list(df.columns) == ['mid', 'spread']
        assert len(df.index) == 100
        assert df.attrs['base'] == self.base.value
        assert df.attrs['quote'] == self.quote.value
        
    def test_data_prepocess(self, mocker: MockerFixture) -> NoReturn:
        mocker.patch(
            'src.data.utils.dd.read_csv',
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
