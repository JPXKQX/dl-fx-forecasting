from dataclasses import dataclass
from typing import Tuple, Callable
from datetime import datetime
from src.data import constants
from src.data.constants import Currency

import dask.dataframe as dd
import os


@dataclass
class DataLoader:
    base: Currency
    quote: Currency
    dir: str = constants.WORKING_DIR + "/data/raw/"
    
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
        period: Tuple[str, str] = None,
        agg: Callable = None
    ) -> dd.DataFrame:
        folder, to_invert = self._search_pair()
        filter_dates = None
        if period is not None:
            start = datetime.strptime(period[0], "%Y-%m-%d")
            end = datetime.strptime(period[1], "%Y-%m-%d")
            filter_dates = [('time', '>=', start), ('time', '<=', end)]
        
        # Read data
        df = dd.read_parquet(folder, filters = filter_dates)

        # Preprocess
        df.attrs = {'base': self.base.value, 
                    'quote': self.quote.value}
        df = df.set_index('time')
        
        # Invert in case of pairs were asked reversed.
        if to_invert: df = df.rdiv(1, fill_value=None)
        
        # Aggregate if it is asked.
        if agg:
            return df.agg()
        return df