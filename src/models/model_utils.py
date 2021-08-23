from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from src.models.sklearn_models import LinearRegr, ElasticNetRegr
from src.models.neural_network import MultiLayerPerceptron
from src.models.inception_time import InceptionTime
from src.models.rnn import LongShortTermMemory

from pathlib import Path
from typing import Union, Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
import logging
import yaml


log = logging.getLogger("Model utilities")


# Dictionary with the conversion from string to object
str2model = {
    'LinearRegression': LinearRegr,
    'ElasticNet': ElasticNetRegr,
    'MultiLayerPerceptron': MultiLayerPerceptron,
    'LSTM': LongShortTermMemory,
    'RandomForest': RandomForestRegressor,
    'InceptionTime': InceptionTime
}


def read_yaml_models(filename: Union[str, Path]) -> Dict:
    log.debug(f"Loading YAML file at {filename}")
    with open(filename, 'r') as stream:
        try:
            json = yaml.safe_load(stream)
            log.debug(f"YAML file at {filename} loaded succesfully.")
        except yaml.YAMLError as e:
            log.error(e)
            raise e

    for key, value in json.items():
        if 'model' in value:
            try:
                json[key]['model'] = str2model[value['model']]
                log.debug(f"Model {value['model']} was obtained successfully.")
            except KeyError as e:
                log.error(e)
                raise e
    return json


def generate_search_space(params: Dict) -> Tuple[Dict, int]:
    num_samples = 10
    for attr, values in params.items():
        if attr == 'num_samples':
            num_samples = int(values)
        elif isinstance(values, list) and len(values) > 1:
            if isinstance(values[0], str) and (not isinstance(values[1], str)):
                params[attr] = getattr(tune, values[0])(*values[1:])
            else:
                params[attr] = tune.choice(values)
        elif isinstance(values, list):
            params[attr] = tune.choice(values)
        else:
            params[attr] = tune.choice([values])
    params.pop('num_samples', None)
    return params, num_samples


def evaluate_predictions(model, features, labels, model_type: str = 'regression'):
    preds = model.predict(features)
    if model_type == 'regression':
        return evaluate_predictions_regression(preds, labels)
    elif model_type == 'classification':
        labels = pd.get_dummies(labels.iloc[:, 0])
        return evaluate_predictions_classification(preds, labels)
    else:
        raise Exception(f"Model type \'{type}\' not valid for evaluation.")


def evaluate_predictions_regression(
    predictions: Union[pd.DataFrame, np.ndarray],
    labels: Union[pd.DataFrame, np.ndarray]
) -> Tuple[Dict, ...]:
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.values
    log.debug(f"Computing test metrics for a total of {len(labels)} instances.")
    vals = {
        'exp_var': float(metrics.explained_variance_score(labels, predictions)),
        'maxerr': float(metrics.max_error(labels, predictions)),
        'mae': float(metrics.mean_absolute_error(labels, predictions)),
        'mse': float(metrics.mean_squared_error(labels, predictions)),
        'r2': float(metrics.r2_score(labels, predictions))
    }
    ssd = ((labels - predictions.reshape(-1, 1)) ** 2).cumsum()
    sst = (labels ** 2).cumsum()
    r2time = (sst - ssd) / sst.iloc[-1]
    return vals, r2time


def evaluate_predictions_classification(
    predictions: pd.DataFrame, 
    labels: pd.DataFrame
) -> Tuple[Dict, None]:
    log_loss = metrics.log_loss(labels, predictions)
    labels = labels.idxmax(axis=1).astype(int)
    predictions = predictions.idxmax(axis=1).astype(int)
    log.info(metrics.classification_report(labels, predictions))
    dict_metrics = metrics.classification_report(labels, predictions, output_dict=True)
    dict_metrics['loss(not weighted)'] = log_loss
    return dict_metrics, None


def init_scheduler_and_search_algorithms(
    search_space: Dict,
    init_config: List[Dict] = None
) -> Tuple[AsyncHyperBandScheduler, ConcurrencyLimiter]:
    # Use HyperBand scheduler to earlystop unpromising runs
    scheduler = AsyncHyperBandScheduler(
        time_attr='training_iteration', metric="val_loss", mode="min", grace_period=10
    )

    # Use bayesian optimisation with TPE implemented by hyperopt
    search_alg = HyperOptSearch(
        search_space, metric="val_loss", mode="min", points_to_evaluate=init_config
    )

    # We limit concurrent trials to 2 since bayesian optimisation doesn't parallelize very well
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
    
    return scheduler, search_alg
