from src.data.constants import Currency, ROOT_DIR
from src.models.model_selection import ModelTrainer
from src.models import model_utils
from typing import Tuple, List, NoReturn, Union

import logging


log = logging.getLogger("Model trainer")


def train_regressions_features(
    base: Currency,
    quote: Currency,
    models_file: str,
    freqs: List[int],
    future_obs: Union[int, List[int]],
    train_period: Tuple[str, str], 
    test_period: Tuple[str, str],
    aux_pair: Tuple[Currency, Currency] = None,
    models_path: str = ROOT_DIR + "/models/configurations/"
) -> NoReturn:
    if isinstance(future_obs, int):
        future_obs = [future_obs]

    models = model_utils.read_yaml_models(models_path + models_file + ".yaml")

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
    models_file: str,
    past_obs: Union[int, List[int]],
    future_obs: Union[int, List[int]],
    train_period: Tuple[str, str], 
    test_period: Tuple[str, str],
    models_path: str = ROOT_DIR + "/models/configurations/"
) -> NoReturn:
    if isinstance(past_obs, int):
        past_obs = [past_obs]

    if isinstance(future_obs, int):
        future_obs = [future_obs]

    models = model_utils.read_yaml_models(models_path + models_file + ".yaml")

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
        level=logging.DEBUG, 
        filename=ROOT_DIR + "/logs/training.log",
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    train_period = '2020-04-06', '2020-04-11'
    test_period = '2020-04-12', '2020-04-18'
    freqs = [1, 2, 3, 5, 10, 25, 50, 100, 200]
    
    train_regressions_features(
        Currency.EUR, Currency.GBP, "initial_models", freqs, [10, 20], 
        train_period, test_period, (Currency.GBP, Currency.JPY))
    #train_regressions_features(
    #    Currency.EUR, Currency.USD, models, freqs, [5, 10, 20], 
    #    train_period, test_period, (Currency.EUR, Currency.JPY))
    #train_regressions_features(
    #    Currency.GBP, Currency.USD, models, freqs, [5, 10, 20], 
    #    train_period, test_period, (Currency.USD, Currency.MXN))