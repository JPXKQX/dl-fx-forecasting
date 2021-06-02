from src.data.constants import Currency
from src.data.data_loader import DataLoader
from src.data import utils, constants
from typing import Tuple, Union, NoReturn
from dataclasses import dataclass
from datetime import datetime

import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import logging
import math


log = logging.getLogger("Line plotter")


@dataclass
class PlotCDFCurrencySpread:
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

    def plot_cdf(self, df: pd.DataFrame, date: str) -> NoReturn:
        label_ticks = self.tick_size()
        labels = [f"{self.base.value}/{self.quote.value}"]
        fig = ff.create_distplot([self.ticks_augment * df['spread']],
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
        df = dl.read(period)
        date_string = utils.period2str(period)
        self.plot_cdf(df, date_string)


@dataclass
class PlotStatsCurrencySpread:
    base: Currency
    quote: Currency
    agg_frame: str = 'D'
    ticks_augment: int = 1000
    path: str = "data/raw/"

    def tick_size(self):
        tick_pow = math.log10(self.ticks_augment)
        if tick_pow.is_integer():
            return f"Sample size (10<sup>{-tick_pow:.0f}</sup> ticks)"
        else:
            raise ValueError(f"Please select a tick augment multiple of 10.")

    def plot_boxplot(
        self, 
        df: pd.DataFrame, 
        title_spec: Tuple[str, str]
    ) -> NoReturn:
        label_ticks = self.tick_size()
        title = f"{title_spec[0]} spread of {self.base.value}/{self.quote.value} {title_spec[1]}"
                
        fig = go.Figure()
    
        for stat in df.columns:
            values = df[stat].values
            fig.add_trace(go.Box(x=self.ticks_augment * values, 
                                 name=constants.stat2label[str(stat)]))
        fig.update_layout(
            title={
                'text': title,
                "x": 0.05,
                "y": 0.95,
                "xanchor": "left",
                "yanchor": "top"
            },
            legend_title=f"{title_spec[0]} Statistic",
            font=dict(
                family="Courier New, monospace",
                size=18
            ))
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(
            title_text=label_ticks,
            title_standoff = 15,
            title_font={"family": "Courier New, monospace", "size": 20})
        fig.show()

    def run(
        self,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None, 
        include_max: bool = False
    ) -> NoReturn:
        # Get the data
        dl = DataLoader(self.base, self.quote, self.path)
        df = dl.read(period)
        statistics = ['std', 'max', 'mean', 'min']
        grouper, freq = utils.filter_datetime_series(df.index, self.agg_frame)
        main_stats = df['spread'].groupby(grouper).aggregate(statistics)
        quantiles = df['spread'].groupby(grouper).quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        stats = pd.concat([main_stats, quantiles.unstack()], axis=1)
        idx = [0, 1, -1, -2, 2, -3, -4, -5, 3]
        if not include_max: idx.remove(1)
        self.plot_boxplot(stats.iloc[:, idx], [freq, utils.period2str(period)])
