from sklearn import metrics

import logging


log = logging.getLogger("Model utilities")


def evaluate_predictions(model, features, labels):
    log.debug(f"Computing test metrics for a total of {len(labels)} instances.")
    preds = model.predict(features)
    exp_var = float(metrics.explained_variance_score(labels, preds))
    maxerr = float(metrics.max_error(labels, preds))
    mae = float(metrics.mean_absolute_error(labels, preds))
    mse = float(metrics.mean_squared_error(labels, preds))
    r2 = float(metrics.r2_score(labels, preds))
    return exp_var, maxerr, mae, mse, r2