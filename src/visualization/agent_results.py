import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from src.data.constants import Currency, ROOT_DIR
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


def load_eur_vs(currency: Currency, scaling_difficulty: str):
    dir = Path(ROOT_DIR) / "models" / "agent" / scaling_difficulty / \
          f"run_EUR{currency.value}_1000steps"
    agent_df = pd.read_csv(dir / f"agent_results_{scaling_difficulty}.csv")
    models_df = pd.read_csv(dir / f"models_results_{scaling_difficulty}.csv")
    losses = pd.read_csv(dir / f"{scaling_difficulty}_training_losses.csv")

    return agent_df, models_df, losses


def plot_differenc_comparison(difference_series: pd.Series, burn_in: int = 0):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    sns.histplot(difference_series[burn_in:100+burn_in], ax=axs[0], kde=True)
    axs[0].axvline(difference_series[burn_in:100+burn_in].mean(), ls='--', c='k')
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_ylabel("Count", fontsize='x-large')
    axs[0].set_xlabel("Agent's improvement (in pips)", fontsize='x-large')
    sns.histplot(difference_series[-100:], ax=axs[1], kde=True)
    axs[1].axvline(difference_series[-100:].mean(), ls='--', c='k')
    axs[1].set_xlabel("Agent's improvement (in pips)", fontsize='x-large')
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_training_loss(loss: pd.Series):
    plt.figure(figsize=(6, 3))
    loss.index = loss.index / 1e3
    loss.rolling(2000).mean().plot()
    plt.ylabel("Q-Network loss")
    plt.legend('', frameon=False)
    plt.xlabel("Episode")
    plt.tight_layout()
    plt.show()


def get_table_model_results(df):
    columns = ['Elastic Net', 'MLP', 'RF']
    indices = [
        'PnL', 'Longs(Number)', 'Longs(PnL)', 'Success Longs(Number)',
        'Success Longs(Pips)', 'Fail Longs(Number)', 'Fails Longs(Pips)',
        'Shorts(Number)', 'Shorts(Pips)', 'Success Shorts(Number)',
        'Success Shorts(Pips)', 'Fail Shorts(Number)', 'Fails Shorts(Pips)'
    ]
    res = pd.DataFrame(df.mean().values.reshape(3, -1).T, columns=columns, index=indices)
    return print(res.to_latex(float_format="%.4f", escape=False))


if __name__ == '__main__':
    agent_df, model_df, losses = load_eur_vs(Currency.USD, 'frictionless')
    get_table_model_results(model_df)
