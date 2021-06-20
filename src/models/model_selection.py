from src.features.build_features import FeatureBuilder
from src.data.data_loader import DataLoader
from src.data.constants import Currency, ROOT_DIR
from .model_utils import evaluate_predictions

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from dataclasses import dataclass
from typing import Tuple, Dict, List

import pandas as pd
import yaml
import logging
import os
import pickle


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
    freqs_features: List[int]
    future_obs: int
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    get_features: bool = True
    aux_pair: Tuple[Currency, Currency] = None

    def __post_init__(self):
        # Get fold to cross validation
        self.tscv = TimeSeriesSplit(n_splits=5)
        if self.get_features:
            log.debug("Loading instances using feature selection.")
            fb = FeatureBuilder(self.base, self.quote)
            self.X_train, self.y_train = fb.build(
                self.freqs_features, self.future_obs, self.train_period,
                self.aux_pair)
            self.X_test, self.y_test = fb.build(
                self.freqs_features, self.future_obs, self.test_period,
                self.aux_pair)
        else:
            log.debug("Loading the instances using all past increments.")
            dl = DataLoader(self.base, self.quote)
            self.X_train, self.y_train = dl.load_dataset(
                'linspace', max(self.freqs_features), self.future_obs,  
                self.train_period, is_pandas=True, overlapping=False)
            self.X_test, self.y_test = dl.load_dataset(
                'linspace', max(self.freqs_features), self.future_obs, 
                self.test_period, is_pandas=True, overlapping=False)
        log.info(f"Train({self.X_train.shape[0]}) and test"
                 f"({self.X_test.shape[0]}) datasets have been loaded")

    def get_num_prev_obs(self) -> int:
        if self.get_features:
            return map(self.freqs_features)
        else:
            return self.freqs_features

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
        n_prev_obs = self.get_num_prev_obs()
        results.to_csv(f"{path}model_sel_{model_name}_{pair}_{n_prev_obs}-"
                       f"{self.future_obs}_{train_date}.csv")
        return clf.best_estimator_

    def save_model_results(self, model, name_model):
        log.debug(f"Obtaining results of {name_model}.")
        exp_var, maxerr, mae, mse, r2 = evaluate_predictions(model, self.X_test, 
                                                             self.y_test)
        train_date = "-".join(map(lambda x: x.replace("-", ""), self.train_period))
        test_date = "-".join(map(lambda x: x.replace("-", ""), self.test_period))
        n_prev_obs = self.get_num_prev_obs()
        data = {
            'model': name_model,
            'params': model.get_params(),
            'pair': f"{self.base.value}/{self.quote.value}",
            'samples': 'non-overlapping',
            'past_ticks': n_prev_obs,
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
        if self.get_features: data['features'] = self.freqs_features
        pair = self.base.value + self.quote.value
        filename = f'{name_model}_{self.base.value}{self.quote.value}_' \
                   f'{n_prev_obs}-{self.future_obs}_{train_date}'
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
