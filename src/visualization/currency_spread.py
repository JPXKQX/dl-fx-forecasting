from src.data.constants import Currency
from src.data.data_loader import DataLoader
from src.data import utils
from typing import Tuple, Union, NoReturn
from dataclasses import dataclass
from datetime import datetime

import plotly.figure_factory as ff
import logging
import math
import dask.dataframe as dd


log = logging.getLogger("Line plotter")


@dataclass
class PlotCurrencySpread:
    base: Currency
    quote: Currency
    ticks_augment: int = 1000
    path: str = "data/raw/"

    def tick_size(self):
        tick_pow = math.log10(self.ticks_augment)
        if tick_pow.is_integer():
            return f"Sample size (10<sup>{-tick_pow:.0f}</sup> ticks)"
        else:
            raise ValueError(f"Please select a tick augment multiple of 10.")

    def plot_cdf(self, df: dd.DataFrame, date: str) -> NoReturn:
        label_ticks = self.tick_size()
        labels = [f"{self.base.value}/{self.quote.value}"]
        fig = ff.create_distplot([self.ticks_augment * df['spread'].compute()],
                                 labels, bin_size=0.02, histnorm='probability')
        fig.update_layout(
            title={
                'text': f"Spread of {self.base.value}/{self.quote.value}{date}",
                "x": 0.05,
                "y": 0.95,
                "xanchor": "left",
                "yanchor": "top"
            },
            legend_title="Currency pair",
            font=dict(
                family="Courier New, monospace",
                size=18
            ))
        fig.update_yaxes(
            title_text="Probability", title_standoff = 20, 
            title_font={"family": "Courier New, monospace", "size": 20})
        fig['layout']['yaxis2'].update(title_text='')
        fig.update_xaxes(
            title_text=label_ticks,
            title_standoff = 15,
            title_font={"family": "Courier New, monospace", "size": 20})
        fig.show()

    def run(
        self,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None
    ) -> NoReturn:
        # Get the data
        dl = DataLoader(self.base, self.quote, self.path)
        df_mid = dl.read(period)
        date_string = utils.period2str(period)
        self.plot_cdf(df_mid, date_string)


if __name__ == '__main__':
    pcs = PlotCurrencySpread(Currency.EUR, Currency.USD)
    pcs.run(('2020-05-17', '2020-05-21'))
