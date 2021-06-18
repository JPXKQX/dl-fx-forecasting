from src.data.constants import ROOT_DIR
from pydantic.dataclasses import dataclass

import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import glob
import pickle
import logging


logger = logging.getLogger("Plotter Coefficients")


@dataclass
class PlotCoefsModel:
    pair: str = "EURUSD"
    models_path: str = ROOT_DIR + "/models/"

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
        paths = glob.glob(self.models_path + f"**/{self.pair}/*.yml")
        for i, path in enumerate(paths):
            with open(path) as results_file:
                results = yaml.load(results_file)
            _, mo, pair, freq, period = path.split("/")[-1].replace(".pkl", 
                                                                "").split("_")
            prev_obs, next_obs = map(int, freq.split("-"))
            logger.info(f"Results loaded for model {mo} for predicting price "
                        f"increment in {pair} at {next_obs} observations ahead,"
                        f"with the last {prev_obs} observations.")


if __name__ == '__main__':
    PlotCoefsModel().get_results()
