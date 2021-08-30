from src.data.constants import ROOT_DIR
from pydantic.dataclasses import dataclass
from typing import List

import os
import seaborn as sns
import numpy as np
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
    variables: List[str] = None
    aux: str = None
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

    def get_results(self):
        paths = glob.glob(f"{self.models_path}**/{self.pair}/{self.aux}/"
                          f"{'_'.join(self.variables)}/*.yml")
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
        fig, axs = plt.subplots(len(metrics), 1, sharex=True, figsize=(12, 9))
        for i, metric in enumerate(metrics):
            sns.lineplot(data=df, x='n_fut', y=metric2label[metric], 
                         hue='model', ax=axs[i], legend=False)

        handles = axs[0].get_lines()
        fig.legend(handles, list(df.model.unique()), loc=(0.8, 0.9))
        axs[0].set_title(self.pair, fontsize=22)
        plt.xlabel("Observations  ahead", fontsize=18)
        plt.show()
        plt.tight_layout()
        print(df)

    def get_coefs(
        self, 
        models: List[str] = ['LinearRegression', 'ElasticNet'],
        obs_aheads: List[str] = [5, 10, 20],
        features: List[str] = [1, 2, 3, 5, 10, 25, 50, 100, 200]
    ):
        coeffs = []
        for i, obs_ahead in enumerate(obs_aheads):
            for j, model_name in enumerate(models):
                path = f"{self.models_path}{model_name}/{self.pair}/" + \
                       f"{self.aux if self.aux else 'raw'}/" + \
                       f"{'_'.join(self.variables)}/*{obs_ahead}*.pkl"
                path = glob.glob(path)[0]
                with open(path, 'rb') as outfile:
                    model = pickle.load(outfile)
                if model_name == 'RandomForest':
                    coeffs.append(model.feature_importances_.reshape((-1, len(features))))
                else:
                    coeffs.append(model.coef_.reshape((-1, len(features))))

        fig, axs = plt.subplots(3, len(models), figsize=(12, 7), sharex=True, sharey=True)
        c_max = max([abs(np.array(coeffs).max()), abs(np.array(coeffs).min())])
        c_min = 0 if np.array(coeffs).min() > 0 else -c_max
        cbar_ax = fig.add_axes([.91, .2, .03, .6])
        for i, ax in enumerate(axs.flat):
            logger.info(f"Plotting heatmap ")
            sns.heatmap(coeffs[i], ax=ax, cmap='RdBu' if c_min < 0 else 'OrRd',
                        vmin=c_min, vmax=c_max, cbar_ax=None if i else cbar_ax,
                        cbar=i == 0, xticklabels=features, yticklabels=[])
            if len(ax.get_yticks() > 1):
                labels = ["EURGBP"] + [self.aux.capitalize() + f"({self.aux})"]
                ax.set_yticklabels(labels, size=16)
            else:
                ax.get_yaxis().set_ticks([])
                ax.get_yaxis().set_ticklabels([])
            ax.set_xticklabels(features, size=16, rotation=0)

        for i, mo in enumerate(models):
            if len(models) > 1:
                axs[0, i].set_title(mo, fontsize=28)
                axs[-1, i].set_xlabel("Span of features", fontsize=22)
            else:
                axs[0].set_title(mo, fontsize=28)
                axs[-1].set_xlabel("Span of features", fontsize=22)
            
        for i, next_o in enumerate(obs_aheads):
            if len(models) > 1:
                axs[i, 0].set_ylabel(f"Horizon: {next_o}", fontsize=22, rotation=90)
            else:
                axs[i].set_ylabel(f"Horizon: {next_o}", fontsize=22, rotation=90)
        
        plt.xticks(rotation=90)
        fig.tight_layout(rect=[0, 0, .9, 1])
        plt.show()


def get_test_results(path):
    with open(path) as results_file:
        results = yaml.load(results_file)
    test_results = results['test']
    if 'explained_variance' in test_results.keys():
        explained_var = test_results['explained_variance']
        maxe = test_results['max_errors']
        mae = test_results['mean_absolute_error']
        mse = test_results['mean_squared_error']
    else:
        explained_var = test_results['exp_var']
        maxe = test_results['maxerr']
        mae = test_results['mae']
        mse = test_results['mse']
    r2 = test_results['r2']
    return explained_var, maxe, mae, mse, r2


def get_table(model: str, target_pair: str):
    path_target = f"{ROOT_DIR}/models/increment/{model}/{target_pair}"

    metrics = ['R2', 'MAE', 'MSE', 'MAXE']
    horizons = [5, 10, 20]
    columns = pd.MultiIndex.from_product(
        [metrics, horizons], names=["metric", "horizon"]
    )

    tuples_multiindex = []
    df = pd.DataFrame(columns=columns)
    for aux in os.listdir(path_target):
        path_aux = path_target + "/" + aux
        for varnames in os.listdir(path_aux):
            _, maxe5, mae5, mse5, r2_5 = get_test_results(
                f"{path_aux}/{varnames}/test_{model}_{target_pair}_200-{5}_20200405-20200411.yml"
            )
            _, maxe10, mae10, mse10, r2_10 = get_test_results(
                f"{path_aux}/{varnames}/test_{model}_{target_pair}_200-{10}_20200405-20200411.yml"
            )
            _, maxe20, mae20, mse20, r2_20 = get_test_results(
                f"{path_aux}/{varnames}/test_{model}_{target_pair}_200-{20}_20200405-20200411.yml"
            )
            df.loc[df.shape[0]] = [r2_5, r2_10, r2_20,
                                   mae5, mae10, mae20,
                                   mse5, mse10, mse20,
                                   maxe5, maxe10, maxe20]
            tuples_multiindex.append((aux, varnames))

    df.index = pd.MultiIndex.from_tuples(
        tuples_multiindex, names=['Aux Currency', 'Variables']
    )
    return print(df.to_latex(float_format="%.4f", escape=False))


if __name__ == '__main__':
    PlotCoefsModel("EURGBP", "increment", ['increment', 'difference'], "USD").get_coefs(['RandomForest'])
