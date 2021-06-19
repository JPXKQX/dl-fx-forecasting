from src.data import utils, constants
from src.data.constants import Currency
from src.data.data_loader import DataLoader
from dataclasses import dataclass
from typing import List, Tuple, NoReturn
from plotly.subplots import make_subplots
from scipy.signal import correlate
from statsmodels.tsa.stattools import acf

import plotly.graph_objects as go
import numpy as np
import logging


log = logging.getLogger("Heatmap plotter")


@dataclass
class PlotCorrelationHeatmap:
    var: str 
    path: str = f'{constants.ROOT_DIR}/data/raw/'
    agg_by: str = 'H'
    
    def compute_corr_matrix(
        self, 
        period: Tuple[str, str] = None
    ) -> Tuple[np.array, List[str]]:
        fx_pairs = utils.list_all_fx_pairs(self.path)
        fx_correlations = np.diag(np.ones(len(fx_pairs)))
        
        for i, fx_pair1 in enumerate(fx_pairs):
            base1, quote1 = fx_pair1.split('/')
            dl1 = DataLoader(Currency(base1), Currency(quote1), self.path)
            df_mid1 = dl1.read(period)[self.var].resample(self.agg_by).mean()
            
            for j, fx_pair2 in enumerate(fx_pairs[i+1:], start=i+1):
                log.debug(f'Computing correlation between {fx_pair1} and {fx_pair2}')
                base2, quote2 = fx_pair2.split('/')
                dl2 = DataLoader(Currency(base2), Currency(quote2), self.path)
                df_mid2 = dl2.read(period)[self.var].resample(self.agg_by).mean()

                # Compute the corelation between currency1 and currency2.
                corr = df_mid1.corr(df_mid2)
                fx_correlations[i, j] = fx_correlations[j, i] = corr
                
        return fx_correlations, fx_pairs

    def plot_heatmap(
        self, 
        period: Tuple[str, str] = None
    ) -> NoReturn:
        fx_correlations, fx_pairs = self.compute_corr_matrix(period)
        
        # Plot the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=fx_correlations, 
            x=fx_pairs, 
            y=fx_pairs,
            colorscale='RdBu',
            hovertemplate="Correlation: %{z:.4f}",
            colorbar={
                'thickness': 50,
                'tick0': -1, 
                'dtick': 0.2}
        ))
        
        fig.update_layout(
            title={
                'text': f'Correlation of {constants.var2label[self.var]} '
                        f'between currency pairs{utils.period2str(period)}',
                'font_size': 24, 
                'xanchor': 'left'
            }, xaxis=dict(side='top', tickfont_size=18), 
            yaxis=dict(autorange='reversed', tickfont_size=18))
        fig.show()


@dataclass
class PlotACFCurreny:
    currency: Currency
    varname: str = 'increment'
    agg_frame: str = 'S'
    nlags: int = 100
    path: str = f'{constants.ROOT_DIR}/data/raw/'

    def _plot_acf(
        self, 
        fig: go.Figure, 
        base_df, 
        base_curr: Currency, 
        n_subplot: int,
        n_cols: int,
        period: Tuple[str, str] = None
    ) -> go.Figure:
        df_c = DataLoader(self.currency, base_curr, self.path).read(period)
        pair = df_c.attrs['base'] + "/" + df_c.attrs['quote']
        df_c = df_c[self.varname].resample(self.agg_frame).mean().dropna()
        data = base_df.merge(df_c, left_index=True, right_index=True)
        values = correlate(data.iloc[:, 0], data.iloc[:, 1])
        idx = np.arange(-self.nlags // 2, self.nlags // 2 + 1)
        l = len(values) // 2
        y = values[l-self.nlags//2:self.nlags//2-l]
        
        row, col = n_subplot // 2, (n_subplot % 2) +1
        annot_pos = row - 1 if n_cols == 1 else 2 * row + col - 3
        fig['layout'].annotations[annot_pos].update(text=f"with {pair}")
        fig.add_trace(go.Scatter(x=idx, y=y), row=row, col=col)
        return fig

    def run(
        self, 
        base: Currency,
        period: Tuple[str, str] = None
    ) -> NoReturn:
        currencies = utils.list_currencies_against(self.currency)
        if base in currencies:
            currencies.remove(base)
        else:
            raise ValueError("The base currency pair is not considered.")
        
        df = DataLoader(self.currency, base, self.path).read(period)
        ref_pair = df.attrs['base'] + "/" + df.attrs['quote']
        df = df[[self.varname]].resample(self.agg_frame).mean().dropna()

        # Get params of subplots
        specs = None
        if len(currencies) > 4:
            rows, cols = (len(currencies) + 1) // 2, 2
            step_subplots = 1
            if (len(currencies) + 1) % 2 == 1:
                first_subplot = 4
                specs = [[{"colspan": 2}, None], [{}, {}] * len(currencies)]
            else:
                first_subplot = 3
        else:
            rows, cols = len(currencies) + 1, 1
            first_subplot, step_subplots = 2, 2

        # Create figure with one plot for each pair
        fig = make_subplots(rows=rows, cols=cols, specs=specs, 
                            subplot_titles=[' '] * (len(currencies) + 1))

        df_acf = acf(df, nlags=self.nlags)
        fig.add_trace(go.Scatter(x=np.arange(self.nlags+1), y=df_acf), row=1, 
                      col=1)
        fig['layout'].annotations[0].update(text=f"with {ref_pair} (ACF)")
        for i, currency in enumerate(currencies, start=first_subplot):
            fig = self._plot_acf(fig, df, currency, step_subplots * i, cols,
                                 period)

        last_ax = f'xaxis{len(currencies)+1}'
        fig['layout'][last_ax].update(title_text=f'Lag')
        fig.update_layout(showlegend=False)
        fig.update_layout(title={
            "text": f"Cross-Correlation of {ref_pair}{utils.period2str(period)}", 
            "font_size": 28})
        fig.show()
