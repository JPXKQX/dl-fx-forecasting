from src.data.constants import Currency
from src.data.data_loader import DataLoader
from typing import List, Tuple, Union, NoReturn
from plotly.subplots import make_subplots
from dataclasses import dataclass, field

import plotly.graph_objects as go
import numpy as np
import dask.dataframe as dd
import logging


log = logging.getLogger("Line plotter")

freq2label = {
    "D": "Daily",
    "H": "Hourly",
    "M": "Per Minute",
    "S": "Per Second"
}


@dataclass
class PlotCurrencyPair:
    base: Currency
    quote: Currency
    freqs: List[Union[str, None]] = field(default_factory=lambda: ['D', 'H'])
    path: str = "data/raw/"

    def prepare_dataframes(self, df: dd.DataFrame) -> dd.DataFrame:
        dfs = []
        for freq in self.freqs:
            if freq in ['D', 'H', 'M', 'S']:
                log.debug(f"Resampling dataframe averaging each {freq}")
                df_processed = df.resample('D').mean()
                dfs.append(df_processed.compute())
            else:
                log.debug("No resampling is also included.")
                dfs.append(df)
            
        return dfs

    def show_dataframes(self, dfs: List[dd.DataFrame]) -> NoReturn:
        fig = make_subplots(
            rows=len(self.freqs), cols=1, shared_xaxes=True,
            subplot_titles=["Mid prices"] + [""] * len(self.freqs[:-1]), 
            vertical_spacing=0.12)
        fig.update_annotations(font_size=20)

        for i, df in enumerate(dfs, start=1):
            label = freq2label[self.freqs[i-1]] if self.freqs[i-1] else "No"
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df.mid, 
                    mode='lines', 
                    name=label),
                row=i, col=1)
        fig.update_layout(
            title={
                'text': f"{self.base.value}/{self.quote.value}",
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
        fig.update_yaxes(title_text=self.quote.value, 
                        title_font={"family": "Courier New, monospace", "size": 20},
                        title_standoff = 20)
        fig.update_xaxes(title_text="Date", 
                        title_font={"family": "Courier New, monospace", "size": 20},
                        title_standoff = 15)
        fig.show()

    def run(
        self,
        period: Tuple[str, str] = None
    ) -> Tuple[np.array, List[str]]:
        # Get the data
        dl = DataLoader(self.base, self.quote, self.path)
        df_mid = dl.read(period)
        dfs = self.prepare_dataframes(df_mid)
        log.info(f"Data is prepared to be shown.")
        
        # Plot images
        self.show_dataframes(dfs)
