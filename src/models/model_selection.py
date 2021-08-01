from src.features.build_features import FeatureBuilder
from src.data.constants import Currency, ROOT_DIR
from src.models.model_utils import evaluate_predictions, generate_search_space, \
    init_scheduler_and_search_algorithms
from src.models.training_plot_utils import save_r2_time_struc
from ray import tune
from src.models.ray_tuning_models import get_training_ray_method

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from dataclasses import dataclass
from typing import Tuple, Dict, List, Union

import pandas as pd
import yaml
import logging
import os
import dill as pickle


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
        target_var (str): variable to predict. Choices
        freqs_features (int): number of previous observations to consider.
        future_obs (int): horizon length of the prediction.
        train_period (Tuple[str, str]): period of training data.
        test_period (Tuple[str, str]): period of testing data.
        variables (List[str]): variables containing any of the string passed are
        considered for the modelling.
        vars2drop (List[str]): variables selected containing any of these string are 
        dropped.
        aux_pair (tuple[Currency]): auxiliary currency to include as model features.
        label
    """
    base: Currency
    quote: Currency
    target_var: str
    freqs_features: Union[int,List[int]]
    future_obs: int
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    variables: List[str] = None
    vars2drop: List[str] = None
    aux_pair: Tuple[Currency, ...] = None

    def __post_init__(self):
        # Get fold to cross validation
        self.tscv = TimeSeriesSplit(n_splits=5)
        fb = FeatureBuilder(self.base, self.quote)
        
        self.X_train, self.y_train = fb.build(
            self.freqs_features, self.future_obs, self.target_var, self.train_period,
            self.aux_pair, self.variables, self.vars2drop)
        self.X_test, self.y_test = fb.build(
            self.freqs_features, self.future_obs, self.target_var, self.test_period,
            self.aux_pair, self.variables, self.vars2drop)

        log.info(f"Train({self.X_train.shape[0]}) and test"
                 f"({self.X_test.shape[0]}) datasets have been loaded")

    def get_num_prev_obs(self) -> int:
        if isinstance(self.freqs_features, list):
            return max(self.freqs_features)
        else:
            return self.freqs_features

    def train(self, models: Dict[str, object]):
        for name, model in models.items():
            log.info(f"Training model {name} for period "
                    f"{' '.join(self.train_period)}.")
            if 'ray_params' in model.keys():
                self.tune_and_train_model(model, name)
            elif 'params' in model.keys():
                self.select_and_train_model(model, name)
            else:
                self.train_model(model, name)

    def train_model(self, model, name: str):
        # Train models
        if 'attrs' in model.keys():
            mo = model['model'](**model['attrs'])
        else:
            mo = model['model']()
        mo.fit(self.X_train, self.y_train)
        self.save_model_results(mo, name)

    def select_and_train_model(self, model, name: str):        
        # Model selection & model training
        mo = model['model']()
        best_mo = self.model_selection(mo, name, model['params'])
        if name in ['LinearRegression', 'RandomForest', 'ElasticNet']:
            best_mo.fit(self.X_train, self.y_train)
        else:
            best_mo.fit(self.X_train, self.y_train, validation_split=0)
        self.save_model_results(best_mo, name)

    def tune_and_train_model(self, model, name: str):        
        # Model selection & model training
        params, num_samples = generate_search_space(model['ray_params'])
        log.info("Initializing ray Trainable")
        training = get_training_ray_method(
            model['model'], self.X_train, self.y_train, ROOT_DIR + '/ray_results/'
        )
        scheduler, search_alg = init_scheduler_and_search_algorithms(params)
        log.info("Starting hyperparameter tuning (ray)")
        results = tune.run(
            training,
            name=name,
            verbose=1,
            max_failures=2,
            search_alg=search_alg,
            local_dir=ROOT_DIR + f"/ray_results",
            scheduler=scheduler,
            num_samples=num_samples
        )
        log.debug(results.results_df)
        log.info(f"Best trial results are obtained with configuration: "
                 f"{results.get_best_config(metric='val_loss', mode='min')}")
        log.info(f"Best trial are: "
                 f"{results.get_best_trial(metric='val_loss', mode='min').last_result}")
        best_mo = model['model'](
            **results.get_best_config(metric="val_loss", mode='min')
        )
        best_mo.fit(self.X_train, self.y_train, validation_split=0)
        self.save_model_results(best_mo, name)

    def model_selection(self, model, model_name, params):
        clf = GridSearchCV(model, param_grid=params, cv=self.tscv, verbose=4, 
                           scoring=val_metrics, refit=val_metrics[0])
        clf.fit(self.X_train, self.y_train)

        results = pd.DataFrame(clf.cv_results_)
        train_date = "-".join(map(lambda x: x.replace("-", ""), self.train_period))

        # Save results of model selection.
        path = self.get_output_folder(model_name)
        os.makedirs(path, exist_ok=True)
        n_prev_obs = self.get_num_prev_obs()
        results.to_csv(f"{path}model_sel_{model_name}_{self.base.value}"
                       f"{self.quote.value}_{n_prev_obs}-{self.future_obs}_"
                       f"{train_date}.csv")
        return clf.best_estimator_

    def save_model_results(self, model, name_model):
        log.debug(f"Obtaining results of {name_model}.")
        test_metrics, r2time = evaluate_predictions(model, self.X_test, self.y_test)
        train_date = "-".join(map(lambda x: x.replace("-", ""), self.train_period))
        test_metrics['test_date'] = "-".join(map(
            lambda x: x.replace("-", ""), 
            self.test_period
        ))
        n_prev_obs = self.get_num_prev_obs()
        data = {
            'model': name_model,
            'params': model.get_params(),
            'pair': f"{self.base.value}/{self.quote.value}",
            'samples': 'non-overlapping',
            'past_ticks': n_prev_obs,
            'ticks_ahead': self.future_obs,
            'train_period': train_date,
            'test' : test_metrics
        }
        if isinstance(self.freqs_features, list): 
            data['features'] = self.freqs_features
        if self.variables:
            data['variables'] = self.variables
        if self.vars2drop:
            data['deleted variables'] = self.vars2drop
        filename = f'{name_model}_{self.base.value}{self.quote.value}_' \
                   f'{n_prev_obs}-{self.future_obs}_{train_date}'
        path = self.get_output_folder(name_model)
        os.makedirs(path, exist_ok=True)

        # Save results.
        log.debug(f"Saving result of {name_model} to {path}test_{filename}.yml")
        with open(path + f"test_{filename}.yml", 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        # Save R-Squared with time structure
        if r2time is not None:
            plots_path = path.replace("models", "reports/models") 
            os.makedirs(plots_path, exist_ok=True)
            save_r2_time_struc(r2time, f"{plots_path}plot_r2time_{filename}.png")

        attr = getattr(model, "save", None)
        if callable(attr):            
            log.debug(f"Saving model {name_model} to {path}{filename}.h5")
            model.save(filename, path)
        else:
            # Save evaluated model.
            log.debug(f"Saving model {name_model} to {path}{filename}.pkl")
            with open(path + f"{filename}.pkl", 'wb') as f:
                pickle.dump(model, f)

    def get_output_folder(self, model_name) -> str:
        if self.aux_pair:
            aux = "".join(map(lambda x: x.value, self.aux_pair))
        else:
            aux = "no_aux"
        folder = f"{ROOT_DIR}/models/{self.target_var}/{model_name}/{self.base.value}" \
                 f"{self.quote.value}/{aux}/{'_'.join(self.variables)}/"
        return folder
