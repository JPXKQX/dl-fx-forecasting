from src.data import utils
from src.data.constants import Currency
from src.data.data_loader import DataLoader
from typing import List, Tuple, NoReturn

import plotly.graph_objects as go
import numpy as np


def compute_corr_matrix(
    path: str, 
    period: Tuple[str, str] = None
) -> Tuple[np.array, List[str]]:
    fx_pairs = utils.list_all_fx_pairs(path)
    fx_correlations = np.diag(np.ones(len(fx_pairs)))
    
    for i, fx_pair1 in enumerate(fx_pairs):
        dl1 = DataLoader(Currency(fx_pair1[:3]), Currency(fx_pair1[3:]), path)
        df_mid1 = dl1.read(period)['mid'].resample('H').mean()
        
        for j, fx_pair2 in enumerate(fx_pairs[i+1:], start=i+1):
            dl2 = DataLoader(Currency(fx_pair2[:3]), 
                             Currency(fx_pair2[3:]),
                             path)
            df_mid2 = dl2.read(period)['mid'].resample('H').mean()

            # Compute the corelation between currency1 and currency2.
            corr = df_mid1.corr(df_mid2)
            fx_correlations[i, j] = fx_correlations[j, i] = corr
            
    return fx_correlations, fx_pairs


def construct_correlation_heatmap(
    path: str, 
    period: Tuple[str, str] = None
) -> NoReturn:
    fx_correlations, fx_pairs = compute_corr_matrix(path, period)
    
    # Plot the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=fx_correlations, 
        x=fx_pairs, 
        y=fx_pairs,
        hovertemplate="Correlation: %{z:.4f}",
        colorbar={'thickness': 50}
    ))
    
    fig.update_layout(
        xaxis=dict(side='top', tickfont_size=18), 
        yaxis=dict(autorange='reversed', tickfont_size=18))
    fig.show()


if __name__ == '__main__':
    construct_correlation_heatmap("data/raw/", ('2020-04-01', '2020-06-01'))
