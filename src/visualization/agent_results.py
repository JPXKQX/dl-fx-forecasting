import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Union


def plot_results_agent(df: pd.DataFrame, odir: Union[str, Path] = None, mode: str = ""):
    sns.displot(df.Difference, height=9, aspect=1.33)
    if odir:
        plt.savefig(f"{str(odir)}/{mode}_agent_alpha.png")
    else:
        plt.show()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 4), sharex=True)

    df1 = df[['Agent', 'Market']].rolling(100).mean()
    df1.plot(ax=ax1, title='Returns (Moving Average)')

    df2 = df['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=ax2, title='Agent Outperformance (% of episodes, Moving Average)')

    ax1.set_ylabel("Pips")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.axhline(.5, ls='--', c='k')

    sns.despine()
    fig.tight_layout()

    if odir:
        fig.savefig(f"{str(odir)}/{mode}_performance.png", dpi=300)
    else:
        plt.show()
