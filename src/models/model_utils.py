from sklearn import metrics
from pathlib import Path
from typing import Union, Dict, Tuple

import logging
import yaml

log = logging.getLogger("Model utilities")


def read_yaml(filename: Union[str, Path]) -> Dict:
    log.debug(f"Loading YAML file at {filename}")
    with open(filename, 'r') as stream:
        try:
            json = yaml.safe_load(stream)
            log.debug(f"YAML file at {filename} loaded succesfully.")
            return json
        except yaml.YAMLError as e:
            log.error(e)
            raise e


def evaluate_predictions(model, features, labels) -> tuple[float, ...]:
    log.debug(f"Computing test metrics for a total of {len(labels)} instances.")
    preds = model.predict(features)
    exp_var = float(metrics.explained_variance_score(labels, preds))
    maxerr = float(metrics.max_error(labels, preds))
    mae = float(metrics.mean_absolute_error(labels, preds))
    mse = float(metrics.mean_squared_error(labels, preds))
    r2 = float(metrics.r2_score(labels, preds))
    return exp_var, maxerr, mae, mse, r2