from src.data.constants import Currency
from src.models.model_selection import ModelTrainer
from neural_network import MultiLayerPerceptron
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict, List, NoReturn, Union

import logging


log = logging.getLogger("Model trainer")


def train_regressions_features(
    base: Currency,
    quote: Currency,
    models: Dict,
    freqs: List[int],
    future_obs: Union[int, List[int]],
    train_period: Tuple[str, str], 
    test_period: Tuple[str, str],
    aux_pair: Tuple[Currency, Currency] = None
) -> NoReturn:
    if isinstance(future_obs, int):
        future_obs = [future_obs]

    for n_fut in future_obs:
        log.info(f"Modeling increments in price using last {max(freqs)} "
                 f"observations to forecast the increment in {n_fut} "
                 f"observations ahead.")
        mt = ModelTrainer(base, quote, freqs, n_fut, train_period, 
                          test_period, aux_pair=aux_pair)
        mt.train(models)


def train_regressions_raw_data(
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
            'params': {'alpha': [0, 0.2, 0.4, 0.5, 1],
                       'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}
        }, 'LinearRegression': {
            'model': linear_model.LinearRegression
        }
    }
    train_period = '2020-04-01', '2020-06-01'
    test_period = '2020-06-01', '2020-07-01'
    freqs = [1, 2, 3, 5, 10, 25, 50, 100, 200]
    
    train_regressions_features(
        Currency.EUR, Currency.USD, models, freqs, [5, 10, 20], 
        train_period, test_period, (Currency.EUR, Currency.JPY))
    train_regressions_features(
        Currency.GBP, Currency.USD, models, freqs, [5, 10, 20], 
        train_period, test_period, (Currency.EUR, Currency.USD))