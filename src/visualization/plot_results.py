from src.data.constants import ROOT_DIR
from pydantic.dataclasses import dataclass
from typing import List

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import glob
import pickle
import logging


logger = logging.getLogger("Plotter Coefficients")

metrics = ['explained_var', 'maxe', 'mae', 'mse', 'r2']
metric2label = {
    'explained_var': 'Explained Variance',
    'maxe': 'Maximum Error',
    'mae': 'Mean Absolute Error',
    'mse': 'Mean Squared Error', 
    'r2': 'R-squared'
}

@dataclass
class PlotCoefsModel:
    pair: str
    which: str = None
    models_path: str = ROOT_DIR + "/models/"
    
    def __post_init__(self):
        if self.which:
            self.models_path += f"{self.which}/"

    def plot_linear_regr(self, model):
        paths = glob.glob(self.models_path + f"{model}/{self.pair}/*.pkl")
        fig, axs = plt.subplots(2, len(paths), sharex=True, sharey=True)
        for i, path in enumerate(paths):
            logger.debug(f"Reading from {path}")
            mo, pair, freq, period = path.split("/")[-1].replace(".pkl", 
                                                                "").split("_")
            prev_obs, next_obs = map(int, freq.split("-"))
            with open(path, 'rb') as f:
                model = pickle.load(f)
            coefs_inc = model.coef_[:prev_obs]
            coefs_spreads = model.coef_[prev_obs:]
            sns.heatmap(coefs_inc.reshape((-1, 10)), ax=axs[0, i], cmap="RdBu",
                        vmin=-0.075, vmax=0.075, cbar=(i + 1 == len(paths)))
            sns.heatmap(coefs_spreads.reshape((-1, 10)), ax=axs[1, i], cmap="RdBu",
                        vmin=-0.075, vmax=0.075, cbar=(i + 1 == len(paths)))
            axs[0, i].set_xlabel(f"{next_obs} ahead")
            axs[0, i].xaxis.set_label_position('top') 

        yticks = list(range(0, prev_obs, 10))
        axs[0, 0].set_yticklabels(yticks, rotation=0)
        axs[0, 0].set_ylabel("Coefficient of increments")
        axs[1, 0].set_yticklabels(yticks, rotation=0)
        axs[1, 0].set_ylabel("Coefficient of spreads")
        plt.tight_layout()
        plt.show()

    def get_results(self, metric: str = 'mse'):
        paths = glob.glob(self.models_path + f"**/{self.pair}/*.yml")
        columns = ['model', 'n_past', 'n_fut', *metrics]
                   
        df = pd.DataFrame(columns=columns)
        for i, path in enumerate(paths):
            _, mo, pair, freq, period = path.split("/")[-1].replace(".yml", "").split("_")
            prev_obs, next_obs = map(int, freq.split("-"))
            logger.info(f"Results loaded for model {mo} for predicting price "
                        f"increment in {pair} at {next_obs} observations ahead,"
                        f"with the last {prev_obs} observations.")
            row = pd.Series([mo, prev_obs, next_obs, * get_test_results(path)], 
                            index = df.columns)
            df = df.append(row, ignore_index=True)

        df = df.rename(metric2label, axis=1)
        fig, axs = plt.subplots(len(metrics), 1, sharex=True)
        for i, metric in enumerate(metrics):
            sns.lineplot(data=df, x='n_fut', y=metric2label[metric], 
                         hue='model', ax=axs[i], legend=False)

        handles = axs[0].get_lines()
        fig.legend(handles, list(df.model.unique()), loc=(0.8, 0.9))
        axs[0].set_title(pair + f" ({period})", fontsize=22)
        plt.xlabel("Observations  ahead", fontsize=18)
        plt.show()
        plt.tight_layout()
        print(df)

    def get_coefs(self, models: List[str] = ['LinearRegression', 'ElasticNet']):
        obs_aheads = [5, 10, 20]
        features = [1, 2, 3, 5, 10, 25, 50, 100, 200]

        coeffs = []
        for i, obs_ahead in enumerate(obs_aheads):
            for j, model in enumerate(models):
                path = self.models_path + f"{model}/" + self.pair + \
                    f"/*{obs_ahead}*.pkl"
                path = glob.glob(path)[0]
                with open(path, 'rb') as outfile:
                    model = pickle.load(outfile)
                coeffs.append(model.coef_.reshape((2, -1)))

        fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .2, .03, .6])
        for i, ax in enumerate(axs.flat):
            sns.heatmap(coeffs[i], ax=ax, cmap='RdBu', cbar= i== 0,
                        vmin=-4, vmax=4, cbar_ax=None if i else cbar_ax,
                        yticklabels=["EURGBP", "GBPJPY"], 
                        xticklabels=features)

        for i, mo in enumerate(models):
            axs[0, i].set_title(mo)
            axs[-1, i].set_xlabel("Span of features")
            
        for i, next_o in enumerate(obs_aheads):
            axs[i, 0].set_ylabel(f"Next {next_o} observations")
        
        plt.xticks(rotation=90)
        fig.tight_layout(rect=[0, 0, .9, 1])
        plt.show()


def get_test_results(path):
    with open(path) as results_file:
        results = yaml.load(results_file)
    test_results = results['test']
    explained_var = test_results['explained_variance']
    maxe = test_results['max_errors']
    mae = test_results['mean_absolute_error']
    mse = test_results['mean_squared_error']
    r2 = test_results['r2']
    return explained_var, maxe, mae, mse, r2


if __name__ == '__main__':
    PlotCoefsModel("EURGBP", "increment_and_spread").get_coefs()
