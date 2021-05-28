from src.data.constants import Currency
from src.data.data_loader import DataLoader
from typing import List, Tuple
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_pair_trends(
    base: Currency,
    quote: Currency, 
    period: Tuple[str, str] = None,
    path: str = "data/raw/"
) -> Tuple[np.array, List[str]]:
    dl = DataLoader(base, quote, path)
    df_mid = dl.read(period)
    df_daily = df_mid.resample('D').mean().compute()
    df_hourly = df_mid.resample('H').mean().compute()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Mid prices", ""], 
                        vertical_spacing=0.12)
    fig.update_annotations(font_size=20)

    fig.add_trace(
        go.Scatter(
            x=df_daily.index, y=df_daily.mid, mode='lines', name='Daily'),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df_hourly.index, y=df_hourly.mid, mode='lines', name='Hourly'),
        row=2, col=1)
    fig.update_layout(
        title={
            'text': f"{base.value}/{quote.value}",
            "x": 0.1,
            "y": 0.95,
            "xanchor": "center",
            "yanchor": "top"
        },
        legend_title="Average",
        font=dict(
            family="Courier New, monospace",
            size=18
        ))
    fig.update_yaxes(title_text=quote.value, 
                     title_font={"family": "Courier New, monospace", "size": 20},
                     title_standoff = 20)
    fig.update_xaxes(title_text="Date", 
                     title_font={"family": "Courier New, monospace", "size": 20},
                     title_standoff = 15)
    fig.show()


if __name__ == '__main__':
    plot_pair_trends(Currency.USD, Currency.EUR, ('2020-04-01', '2020-06-01'))
