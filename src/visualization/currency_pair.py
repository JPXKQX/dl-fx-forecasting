from src.data.constants import Currency
from src.data.data_loader import DataLoader
from src.data import utils, constants
from typing import Tuple, Union, NoReturn, Dict
from dataclasses import dataclass
from datetime import datetime
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging
import os
import math


log = logging.getLogger("Pair plotter")


@dataclass
class PlotCDFCurrencyPair:
    base: Currency
    quote: Currency
    which: str
    ticks_augment: int = 1000
    path: str = "data/raw/"

    def tick_size(self, scale_ticks: int = 1):
        tick_pow = math.log10(self.ticks_augment / scale_ticks)
        if tick_pow.is_integer():
            return f"1 pip = 10<sup> {-tick_pow:.0f} </sup>"
        else:
            raise ValueError(f"Please select a tick augment multiple of 10.")

    def plot_cdf(self, df: pd.DataFrame, date: str) -> NoReturn:
        scale_ticks = df.attrs['scale'] if 'scale' in df.attrs else 1
        label_ticks = self.tick_size(scale_ticks)
        labels = [f"{self.base.value}/{self.quote.value}"]
        fig = px.histogram(self.ticks_augment / scale_ticks * df, labels=labels,
                           x=self.which, marginal='violin', nbins=125, 
                           histnorm='probability', opacity=0.75)
        fig.update_layout(
            title={
                'text': f"{constants.var2label[self.which].capitalize()} of "
                        f"{self.base.value}/{self.quote.value}{date}",
                "x": 0.05,
                "y": 0.97,
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
        fig.update_xaxes(
            title_text="Size (in pips)",
            title_standoff = 15,
            title_font={"family": "Courier New, monospace", "size": 20})
        fig['layout']['yaxis2'].update(title_text='')
        fig['layout']['xaxis2'].update(title_text='')
        # TODO: Fix Bug: Extra "\" when including pip size annotation.
        # fig.add_annotation(text=label_ticks, x=1, y=-0.1, xref='paper',
        #                    yref='paper', xanchor='right', yanchor='top',
        #                    font_size=12)
        
        fig.show()

    def plot_hourly_cdf(self, df: pd.DataFrame, date: str) -> NoReturn:
        fig = make_subplots(
            rows=4, cols=6, shared_xaxes=True, shared_yaxes=True)
        scale_ticks = df.attrs['scale'] if 'scale' in df.attrs else 1
        hour = df.index.hour
        df['row'] = hour % 4 + 1
        df['col'] = hour // 4 + 1
        labels = {
            'increment': "Size (in pips)",
            'mid': "Size (in pips)",
            'spread': "Size (in pips)",
            "probability": "Probability"
        }
        fig = px.histogram(self.ticks_augment / scale_ticks * df, x=self.which, 
                           marginal='violin', opacity=0.75, facet_row='row', 
                           facet_col='col', nbins=125, histnorm='probability', 
                           labels=labels)
        fig.update_layout(
            title={
                'text': f"{constants.var2label[self.which].capitalize()} of "
                        f"{self.base.value}/{self.quote.value}{date}",
                "x": 0.05,
                "y": 1,
                "xanchor": "left",
                "yanchor": "top"
            },
            legend_title="Currency pair",
            font=dict(
                family="Courier New, monospace",
                size=18
            ))
        fig.update_yaxes(
            title_standoff = 15, 
            title_font={"family": "Courier New, monospace", "size": 20})       
        fig.update_xaxes(
            title_standoff = 15,
            title_font={"family": "Courier New, monospace", "size": 20})

        tickets = list(map(lambda x: f'{x}H', range(6))) \
            + ['18H - 23H', '12H - 17H', '6H - 11H', '']
        for i in range(len(fig['layout'].annotations)):
            fig['layout'].annotations[i].update(text=tickets[i])

        fig.show()

    def run(
        self,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None, 
        group_hourly: bool = False
    ) -> NoReturn:
        # Get the data
        dl = DataLoader(self.base, self.quote, self.path)
        df = dl.read(period)
        date_string = utils.period2str(period)
        if group_hourly:
            self.plot_hourly_cdf(df, date_string)
        else:
            self.plot_cdf(df, date_string)


@dataclass
class PlotStatsCurrencyPair:
    base: Currency
    quote: Currency
    which: str
    agg_frame: str = 'D'
    ticks_augment: int = 1000
    path: str = "data/raw/"

    def tick_size(self, scale_ticks: int = 1):
        tick_pow = math.log10(self.ticks_augment / scale_ticks)
        if tick_pow.is_integer():
            return f"1 pip = 10<sup>{-tick_pow:.0f}</sup>"
        else:
            raise ValueError(f"Please select a tick augment multiple of 10.")

    def plot_boxplot(
        self, 
        df: pd.DataFrame, 
        title_spec: Tuple[str, str]
    ) -> NoReturn:
        scale_ticks = df.attrs['scale'] if 'scale' in df.attrs else 1
        label_ticks = self.tick_size(scale_ticks)
        title = f"{title_spec[0]} {constants.var2label[self.which]} of " \
                f"{self.base.value}/{self.quote.value} {title_spec[1]}"
                
        fig = go.Figure()
    
        for stat in df.columns:
            values = df[stat].values
            fig.add_trace(go.Box(x=self.ticks_augment / scale_ticks * values, 
                                 name=constants.stat2label[str(stat)]))
        fig.update_layout(
            title={
                'text': title.capitalize(),
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
            title_text="Size (in pips)",
            title_standoff = 15,
            title_font={"family": "Courier New, monospace", "size": 20})
        # TODO: Fix Bug: Extra "\" when including pip size annotation.
        # fig.add_annotation(text=label_ticks, x=1, y=-0.1, xref='paper',
        #                    yref='paper', xanchor='right', yanchor='top',
        #                    font_size=12)
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
        df_gropued = df[self.which].groupby(grouper)
        main_stats = df_gropued.aggregate(statistics)
        quantiles = df_gropued.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        stats = pd.concat([main_stats, quantiles.unstack()], axis=1)
        idx = [0, 1, -1, -2, 2, -3, -4, -5, 3]
        if not include_max: idx.remove(1)
        self.plot_boxplot(stats.iloc[:, idx], [freq, utils.period2str(period)])

    def get_table(
        self,
        period: Tuple[Union[str, datetime], 
                      Union[str, datetime]] = None,
        output_dir: str = "/tmp/"
    ) -> Tuple[Dict, str, str]:
        variables = ['mid', 'spread', 'increment']
        statistics = ['std', 'max', 'mean', 'min']
        quantiles = [0.0005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.9995]
        results = {}
        
        # Load statistics already computed.
        remaining_vars = []
        for var in variables:
            filename = f"{output_dir}hourly_stats_{var}_{self.base.value}" \
                       f"{self.quote.value}.csv"
            if os.path.isfile(filename):
                log.debug(f"Data loaded from files cached at {filename}")
                results[var] = pd.read_csv(filename, index_col=0)
                continue
            else:
                remaining_vars.append(var)
        
        if len(remaining_vars):
            dl = DataLoader(self.base, self.quote, self.path)
            df = dl.read(period)
            for var in remaining_vars:
                filename = f"{output_dir}hourly_stats_{var}_{self.base.value}" \
                           f"{self.quote.value}.csv"
                log.debug(f"Caching {var} statistics at {filename}")
                df_gropued = df[var].groupby(df.index.hour)
                main_stats = df_gropued.aggregate(statistics)
                quantiles_stats = df_gropued.quantile(quantiles).unstack()
                results_df = pd.concat([main_stats, quantiles_stats], axis=1)
                results_df.reindex(
                    ['std', 'min', '0.0005', '0.05', '0.25', '0.5', 'mean',
                     '0.75', '0.95', '0.9995', 'max'], 
                    axis=1, inplace=True)
                results[var] = results_df
                results[var].to_csv(filename)
        return results, utils.period2str(period), \
            self.base.value + self.quote.value
