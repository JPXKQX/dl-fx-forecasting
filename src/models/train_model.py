from src.data.constants import Currency, ROOT_DIR
from src.models.model_selection import ModelTrainer
from src.models import model_utils
from typing import Tuple, List, NoReturn, Union

import logging


log = logging.getLogger("Model trainer")


def train(
    base: Currency,
    quote: Currency,
    target: str,
    models_file: str,
    freqs: List[int],
    future_obs: Union[int, List[int]],
    train_period: Tuple[str, str], 
    test_period: Tuple[str, str],
    aux_pair: Tuple[Currency, ...] = None,
    variables: List[str] = ['increment'],
    variables_to_drop: List[str] = None,
    models_path: str = ROOT_DIR + "/models/configurations/"
) -> NoReturn:
    if isinstance(future_obs, int):
        future_obs = [future_obs]

    models = model_utils.read_yaml_models(models_path + models_file + ".yml")

    for n_fut in future_obs:
        log.info(f"Modeling increments in price using last "
                 f"{max(freqs) if isinstance(freqs, list) else freqs} "
                 f"observations to forecast the increment in {n_fut} "
                 f"observations ahead.")
        mt = ModelTrainer(base, quote, target, freqs, n_fut, train_period, test_period, 
                          variables=variables, vars2drop=variables_to_drop,
                          aux_pair=aux_pair)
        mt.train(models)


if __name__ == '__main__':
    train_period = '2020-04-05', '2020-04-11'
    test_period = '2020-04-12', '2020-04-18'
    freqs = [1, 2, 3, 5, 10, 25, 50, 100, 200]
    future_obs = [5, 10, 20]
    train(
        Currency.EUR, Currency.GBP, 'fixed-time-increment',  "inceptiontime_classification", 
        200, future_obs, train_period, test_period, (Currency.USD, ), 
        variables=['increment', 'difference', 'spread'])
    train(
        Currency.EUR, Currency.GBP, 'fixed-time-increment',  "inceptiontime_classification", 
        200, future_obs, train_period, test_period, (Currency.USD, Currency.GBP), 
        variables=['increment', 'difference', 'spread'])