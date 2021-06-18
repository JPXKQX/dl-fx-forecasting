from src.data.data_loader import DataLoader
from src.data.constants import Currency, ROOT_DIR
from neural_network import MultiLayerPerceptron
from sklearn import linear_model, metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from typing import Tuple, Dict, List, NoReturn, Union

import pandas as pd
import yaml
import os
import pickle
import logging


log = logging.getLogger("Model trainer")


val_metrics = ['neg_mean_squared_error', 'explained_variance', 'max_error', 
               'neg_mean_absolute_error', 'r2']


@dataclass
class ModelTrainer:
    """ Model to implement model selection and training of different models.
    The models passed as parameter, have to implement the following methods:
        - fit(X, y) to train the model.
        - predict(X) to make a prediction for a given input X.
        - get_params() get parameters of a fitted model.
    
    Attrs:
        base (Currency): base currency to train the model.
        quote (Currency): quote currency to train the model.
        past_n_obs (int): number of previous observations to consider.
        future_obs (int): horizon length of the prediction.
        train_period (Tuple[str, str]): period of training data.
        test_period (Tuple[str, str]): period of testing data.
    """
    base: Currency
    quote: Currency    
    past_n_obs: int
    future_obs: int
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]

    def __post_init__(self):
        # Get fold to cross validation
        self.tscv = TimeSeriesSplit(n_splits=5)
        dl = DataLoader(self.base, self.quote)
        self.X_train, self.y_train = dl.load_dataset(
            'linspace', self.past_n_obs, self.future_obs,  self.train_period, 
            is_pandas=True, overlapping=False)
        self.X_test, self.y_test = dl.load_dataset(
            'linspace', self.past_n_obs, self.future_obs, self.test_period, 
            is_pandas=True, overlapping=False)

    def train(self, models: Dict[str, object]):
        for name, model in models.items():
            log.info(f"Training model {name} for period "
                    f"{' '.join(self.train_period)}.")
            if 'params' in model.keys():
                self.select_and_train_model(model, name)
            else:
                self.train_model(model, name)
    def train_model(self, model, name: str):
        # Train models
        mo = model['model']()
        mo = mo.fit(self.X_train, self.y_train)
        self.save_model_results(mo, name)

    def select_and_train_model(self, model, name: str):        
        # Model selection & model training
        mo = model['model']()
        best_mo = self.model_selection(mo, name, model['params'])
        best_mo.fit(self.X_train, self.y_train)
        self.save_model_results(best_mo, name)

    def model_selection(self, model, model_name, params):
        clf = GridSearchCV(model, param_grid=params, cv=self.tscv, verbose=2, 
                           scoring=val_metrics, refit=val_metrics[0], n_jobs=-1)
        clf.fit(self.X_train, self.y_train)

        results = pd.DataFrame(clf.cv_results_)
        train_date = "-".join(map(lambda x: x.replace("-", ""), self.train_period))

        # Save results of model selection.
        pair = f"{self.base.value}{self.quote.value}"
        path = f"{ROOT_DIR}/models/{model_name}/{pair}/"
        os.makedirs(path, exist_ok=True)
        results.to_csv(f"{path}model_sel_{model_name}_{pair}_{self.past_n_obs}-"
                       f"{self.future_obs}_{train_date}.csv")
        return clf.best_estimator_

    def save_model_results(self, model, name_model):
        log.debug(f"Obtaining results of {name_model}.")
        exp_var, maxerr, mae, mse, r2 = evaluate_predictions(model, self.X_test, 
                                                             self.y_test)
        train_date = "-".join(map(lambda x: x.replace("-", ""), self.train_period))
        test_date = "-".join(map(lambda x: x.replace("-", ""), self.test_period))
        data = {
            'model': name_model,
            'params': model.get_params(),
            'pair': f"{self.base.value}/{self.quote.value}",
            'samples': 'non-overlapping',
            'past_ticks': self.past_n_obs,
            'ticks_ahead': self.future_obs,
            'train_period': train_date,
            'test' : {
                'period': test_date,
                'explained_variance': exp_var,
                'max_errors': maxerr,
                'mean_absolute_error': mae,
                'mean_squared_error': mse,
                'r2': r2
            }
        }
        pair = self.base.value + self.quote.value
        filename = f'{name_model}_{self.base.value}{self.quote.value}_' \
                   f'{self.past_n_obs}-{self.future_obs}_{train_date}'
        path = f"{ROOT_DIR}/models/{name_model}/{pair}/"
        os.makedirs(path, exist_ok=True)

        # Save results.
        log.debug(f"Saving result of {name_model} to {path}test_{filename}.yml")
        with open(path + f"test_{filename}.yml", 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        # Save evaluated model.
        log.debug(f"Saving model {name_model} to {path}{filename}.pkl")
        with open(path + f"{filename}.pkl", 'wb') as f:
            pickle.dump(model, f)


def evaluate_predictions(model, features, labels):
    log.debug(f"Computing test metrics for a total of {len(labels)} instances.")
    preds = model.predict(features)
    exp_var = float(metrics.explained_variance_score(labels, preds))
    maxerr = float(metrics.max_error(labels, preds))
    mae = float(metrics.mean_absolute_error(labels, preds))
    mse = float(metrics.mean_squared_error(labels, preds))
    r2 = float(metrics.r2_score(labels, preds))
    return exp_var, maxerr, mae, mse, r2


def train_regressions(
    base: Currency,
    quote: Currency,
    models: Dict,
    past_obs: Union[int, List[int]],
    future_obs: Union[int, List[int]],
    train_period: Tuple[str, str], 
    test_period: Tuple[str, str]
) -> NoReturn:
    if isinstance(past_obs, int):
        past_obs = [past_obs]

    if isinstance(future_obs, int):
        future_obs = [future_obs]

    for n_past in past_obs:
        for n_fut in future_obs:
            log.info(f"Modeling increments in price using last {n_past} "
                     f"observations to forecast the increment in {n_fut} "
                     f"observations ahead.")
            mt = ModelTrainer(base, quote, n_past, n_fut, train_period, 
                              test_period)
            mt.train(models)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    train_period = '2020-04-01', '2020-06-01'
    test_period = '2020-06-01', '2020-07-01'


    models = {
        'MultiLayerPerceptron': {
            'model': MultiLayerPerceptron,
            'params': {
                'n_neurons': [
                    (32, 64, 128, 64, 32), 
                    (32, 64, 32), 
                    (32, 128, 32)],
                'f_act': ['relu', 'sigmoid', 'tanh']
            }
        }, 'RandomForest': {
            'model': RandomForestRegressor,
            'params': {
                'n_estimators': [75, 100, 150, 200],
                'max_depth': [6, 10, 15, 20],
                'min_samples_leaf': [35, 50, 75, 100]
            }
        }, 'ElasticNet': {
            'model': linear_model.ElasticNet, 
            'params': {'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                       'l1_ratio': [0, 0.10, 0.15, 0.25, 0.3]}
        }, 'LinearRegression': {
            'model': linear_model.LinearRegression
        }
    }
    train_regressions(Currency.EUR, Currency.USD, models, 100, [5, 10, 20], train_period, test_period)
    train_regressions(Currency.GBP, Currency.USD, models, 100, [5, 10, 20], train_period, test_period)