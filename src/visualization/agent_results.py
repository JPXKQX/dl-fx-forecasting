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

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

    df1 = df[['Agent', 'Market']].rolling(100).mean()
    df1.plot(ax=ax1, title='Returns (Moving Average)')

    df2 = (df[['Agent', 'Difference']] > 0).rolling(100).mean()
    df2.plot(ax=ax2, title='Agent Outperformance')

    ax1.set_ylabel("Pips")
    ax1.set_xlabel("Episode")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.axhline(0, ls='--', c='k')
    ax2.set_ylabel("% of episodes")
    ax2.set_xlabel("Episode")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.axhline(.5, ls='--', c='k')
    ax2.legend(["vs Not trading", "vs Market"])

    sns.despine()
    fig.tight_layout()

    if odir:
        fig.savefig(f"{str(odir)}/{mode}_performance.png", dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv("~/Descargas/results_real_world_700.csv")
    plot_results_agent(df)
