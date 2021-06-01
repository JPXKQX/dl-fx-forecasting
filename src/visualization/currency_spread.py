from src.data.constants import Currency
from src.data.data_loader import DataLoader
from typing import List, Tuple, Union
from dataclasses import dataclass

import plotly.figure_factory as ff
import numpy as np
import logging
from datetime import datetime


log = logging.getLogger("Line plotter")


@dataclass
class PlotCurrencySpread:
    base: Currency
    quote: Currency
    path: str = "data/raw/"

    def plot_cdf(self, df):
        labels = [f"Spread {self.base.value}/{self.quote.value}"]
        fig = ff.create_distplot([df['spread'].compute()], labels, bin_size=0.2)
        fig.show()

    def run(
        self,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None
    ) -> Tuple[np.array, List[str]]:
        # Get the data
        dl = DataLoader(self.base, self.quote, self.path)
        df_mid = dl.read(period)
        self.plot_cdf(df_mid)


if __name__ == '__main__':
    pcs = PlotCurrencySpread(Currency.EUR, Currency.USD)
    pcs.run(('2020-04-01', '2020-06-01'))
