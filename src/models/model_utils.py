from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from neural_network import MultiLayerPerceptron
from pathlib import Path
from typing import Union, Dict

import logging
import yaml


# Dictionary with the conversion from string to object
str2model = {
    'LinearRegression': linear_model.LinearRegression,
    'ElasticNet': linear_model.ElasticNet,
    'MultiLayerPerceptron': MultiLayerPerceptron,
    'RandomForest': RandomForestRegressor
}


log = logging.getLogger("Model utilities")


def read_yaml_models(filename: Union[str, Path]) -> Dict:
    log.debug(f"Loading YAML file at {filename}")
    with open(filename, 'r') as stream:
        try:
            json = yaml.safe_load(stream)
            log.debug(f"YAML file at {filename} loaded succesfully.")
        except yaml.YAMLError as e:
            log.error(e)
            raise e
    
    for key, value in json.item():
        if 'model' in value:
            try:
                json[key]['model'] = str2model(value['model'])
                log.debug(f"Model {value['model']} was obtained successfully.")
            except KeyError as e:
                log.error(e)
                raise e
    return json


def evaluate_predictions(model, features, labels) -> tuple[float, ...]:
    log.debug(f"Computing test metrics for a total of {len(labels)} instances.")
    preds = model.predict(features)
    exp_var = float(metrics.explained_variance_score(labels, preds))
    maxerr = float(metrics.max_error(labels, preds))
    mae = float(metrics.mean_absolute_error(labels, preds))
    mse = float(metrics.mean_squared_error(labels, preds))
    r2 = float(metrics.r2_score(labels, preds))
    return exp_var, maxerr, mae, mse, r2