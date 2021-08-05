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

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

    df1 = df[['Agent', 'Market']].sub(1).rolling(100).mean()
    df1.plot(ax=ax1, title='Returns (Moving Average)')

    df2 = df['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=ax2, title='Agent Outperformance (%, Moving Average)')

    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    ax2.axhline(.5, ls='--', c='k')

    sns.despine()
    fig.tight_layout()

    if odir:
        fig.savefig(f"{str(odir)}/{mode}_performance.png", dpi=300)
    else:
        plt.show()
