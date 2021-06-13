from src.data.data_loader import DataLoader
from src.data.constants import Currency, ROOT_DIR
from sklearn import linear_model, metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd
import yaml
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

    def train_model(self, models: Dict[str, object]):
        for name, model in models.items():
            # Train models
            log.info(f"Training model {name} for period "
                     f"{' '.join(self.train_period)}.")
            mo = model['model']()
            mo = mo.fit(self.X_train, self.y_train)
            self.save_model_results(mo, name)

    def select_and_train_model(self, models: Dict[str, object]):        
        for name, model in models.items():
            log.info(f"Training model {name} for period "
                     f"{' '.join(self.train_period)}.")
            # Model selection & model training
            mo = model['model']()
            best_mo = self.model_selection(mo, name, model['params'])
            self.save_model_results(best_mo, name)

    def model_selection(self, model, model_name, params):
        clf = GridSearchCV(model, param_grid=params, cv=self.tscv, verbose=2, 
                           scoring=val_metrics, refit=val_metrics[0], n_jobs=-1)
        clf.fit(self.X_train, self.y_train)

        results = pd.DataFrame(clf.cv_results_)
        train_date = "-".join(map(lambda x: x.replace("-", ""), self.train_period))

        # Save results of model selection.
        results.to_csv(f"{ROOT_DIR}/models/{model_name}/model_sel_{model_name}_"
                       f"{self.base.value}{self.quote.value}_{self.past_n_obs}"
                       f"-{self.future_obs}_{train_date}.csv")
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
        filename = f'{name_model}_{self.base.value}{self.quote.value}_' \
                   f'{self.past_n_obs}-{self.future_obs}_{train_date}'
        path_results = f"{ROOT_DIR}/models/{name_model}/test_{filename}.yml"
        path_model = f"{ROOT_DIR}/models/{name_model}/{filename}.pkl"
        
        # Save results.
        log.debug(f"Saving result of {name_model} to {path_results}.")
        with open(path_results, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        # Save evaluated model.
        log.debug(f"Saving model {name_model} to {path_model}.")
        with open(path_model, 'wb') as f:
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


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    mt = ModelTrainer(
        Currency.EUR, Currency.GBP, 100, 5, ('2020-04-01', '2020-05-01'), 
        ('2020-05-01', '2020-05-10'))

    mt.train_model(
        {'LinearRegression': dict(model=linear_model.LinearRegression)})
    mt.select_and_train_model({
        'ElasticNet': {
            'model': linear_model.ElasticNet, 
            'params': {'alpha': [0, 0.5, 0.8, 1.0, 1.2],
                       'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}}
    })