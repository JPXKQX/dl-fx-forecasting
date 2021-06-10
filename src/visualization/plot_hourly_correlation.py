from src.data import utils, constants
from src.data.constants import Currency
from src.data.data_loader import DataLoader
from dataclasses import dataclass
from typing import List, Tuple, NoReturn

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
