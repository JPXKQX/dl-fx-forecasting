from dataclasses import dataclass
from typing import Tuple, Callable, Union
from datetime import datetime
from src.data import constants, utils
from src.data.constants import Currency

import dask.dataframe as dd
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
    ) -> dd.DataFrame:
        folder, to_invert = self._search_pair()
        filter_dates = None
        if period is not None:
            start = utils.str2datetime(period[0])
            end = utils.str2datetime(period[1])
            filter_dates = [('time', '>=', start), ('time', '<=', end)]
        
        # Read data
        df = dd.read_parquet(folder, filters = filter_dates, 
                             engine="pyarrow-dataset")

        if len(df.index) == 0: 
            raise Exception("No data has been found. The period set may "
                            "correspond to a noon-labor day.")

        # Preprocess
        df = df.set_index('time')
        if to_invert: df = df.rdiv(1, fill_value=None)
        
        df.attrs = {'base': self.base.value, 'quote': self.quote.value}
        return df

