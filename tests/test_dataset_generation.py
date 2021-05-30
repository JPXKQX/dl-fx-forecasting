from src.data.constants import ROOT_DIR, Currency
from src.data.data_extract import DataExtractor
from src.data.data_loader import DataLoader
from datetime import datetime

import pandas as pd


def mock_parquet():
    df = pd.read_csv(f"{ROOT_DIR}/tests/data/EURUSD-parquet.csv", index_col=0)
    return df


class TestDatasetGeneration:
    def prepare(self):
        self.base = Currency.USD
        self.quote = Currency.EUR
        self.path = f"{ROOT_DIR}/data/raw/"
        
    def test_data_loading(self, mocker):
        mocker.patch(
            'src.data.data_loader.dd.read_parquet',
            return_value=mock_parquet()
        )
        self.prepare()
        dl = DataLoader(self.base, self.quote, self.path)
        df = dl.read((datetime(2020, 6, 10), datetime(2020, 6, 20)))
        assert df.columns == ['low', 'high', 'mid']
        assert len(df.index) == 1000
        assert df.attrs['base'] == self.base.value
        assert df.attrs['quote'] == self.quote.value
        
    def test_data_extraction(self):
        self.prepare()
        de = DataExtractor((self.base, self.quote), [5, 6], 2021)
        de.prepare()
