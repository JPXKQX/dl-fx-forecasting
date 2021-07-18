from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from src.models.sklearn_models import LinearRegr, ElasticNetRegr
from src.models.neural_network import MultiLayerPerceptron
from src.models.inception_time import InceptionTime
from pathlib import Path
from typing import Union, Dict, NoReturn

import matplotlib.pyplot as plt
import logging
import pandas as pd
import logging
import yaml


# Dictionary with the conversion from string to object
str2model = {
    'LinearRegression': LinearRegr,
    'ElasticNet': ElasticNetRegr,
    'MultiLayerPerceptron': MultiLayerPerceptron,
    'RandomForest': RandomForestRegressor,
    'InceptionTime': InceptionTime
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
    
    for key, value in json.items():
        if 'model' in value:
            try:
                json[key]['model'] = str2model[value['model']]
                log.debug(f"Model {value['model']} was obtained successfully.")
            except KeyError as e:
                log.error(e)
                raise e
    return json


def evaluate_predictions(model, features, labels, type: str = 'regression'):
    preds = model.predict(features)
    if model.problem == 'regression':
        return evaluate_predictions_regression(preds, labels)
    elif model.problem == 'classification':
        labels = pd.get_dummies(labels.iloc[:, 0])
        return evaluate_predictions_classification(preds, labels)
    else:
        raise Exception(f"Model type \'{type}\' not valid for evaluation.")


def evaluate_predictions_regression(
    predictions: pd.DataFrame, 
    labels: pd.DataFrame
) -> tuple[Dict, ...]:
    log.debug(f"Computing test metrics for a total of {len(labels)} instances.")
    vals = {}
    vals['exp_var'] = float(metrics.explained_variance_score(labels, predictions))
    vals['maxerr'] = float(metrics.max_error(labels, predictions))
    vals['mae'] = float(metrics.mean_absolute_error(labels, predictions))
    vals['mse'] = float(metrics.mean_squared_error(labels, predictions))
    vals['r2'] = float(metrics.r2_score(labels, predictions))
    ssd = ((labels - predictions.reshape(-1, 1)) ** 2).cumsum()
    sst = (labels ** 2).cumsum()
    r2time = (sst - ssd) / sst.iloc[-1]
    return vals, r2time


def evaluate_predictions_classification(
    predictions: pd.DataFrame, 
    labels: pd.DataFrame
) -> tuple[Dict, None]:
    labels = labels.idxmax(axis=1).astype(int)
    predictions = predictions.idxmax(axis=1).astype(int)
    log.info(metrics.classification_report(labels, predictions))
    return metrics.classification_report(labels, predictions, output_dict=True), None


def save_r2_time_struc(r2time, outfile: Union[str, Path]) -> NoReturn:
    log.info(f"Plotting R2 with time structure to {outfile}")
    plt.figure(figsize=(12, 9))
    r2time.plot(legend=False)
    plt.ylabel("R-Squared with time structure")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outfile)
    