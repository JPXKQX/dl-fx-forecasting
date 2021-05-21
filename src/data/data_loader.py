from dataclasses import dataclass
from typing import Tuple
from datetime import datetime
from src.data import constants
from src.data.constants import Currency

import dask.dataframe as dd


@dataclass
class DataLoader:
    base_currency: Currency
    quote_currency: Currency
    dir: str = constants.WORKING_DIR + "/data/raw/"
    
    def read(self, period: Tuple[str, str] = None) -> dd.DataFrame:
        folder = self.dir + self.base_currency.value + self.quote_currency.value
        filter_dates = None
        if period is not None:
            start = datetime.strptime(period[0], "%Y-%m-%d")
            end = datetime.strptime(period[1], "%Y-%m-%d")
            filter_dates = [('time', '>=', start), ('time', '<=', end)]
        
        df = dd.read_parquet(folder, filters = filter_dates)
        return df
    