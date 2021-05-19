from src.data.data_extract import DataExtractor
from src.data.data_preprocess import DataPreprocessor
from src.data.constants import Currency
from typing import NoReturn

import logging


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s | %(levelname)s | %(message)s')


def process_fx_pair(
    currency1: Currency, 
    currency2: Currency, 
    clobber: bool = False
) -> NoReturn:
    """ Process the data corresponding to currency pair specified.

    Args:
        currency1 (Currency): first currency to consider.
        currency2 (Currency): second currency to consider.
        clobber (bool): if true overwrite the data associated.
    """   
    # Unzip data
    dd_2020 = DataExtractor((currency1, currency2), list(range(4, 12)), 2020)
    csv_files_2020 = dd_2020.prepare()
    dd_2021 = DataExtractor((currency1, currency2), [1, 2, 3, 4], 2021)
    csv_files_2021 = dd_2021.prepare() 
    csv_files = csv_files_2020 + csv_files_2021
    
    # Save into Parquet files.
    dp = DataPreprocessor(csv_files)
    dp.save_datasets(clobber) 


if __name__ == '__main__':
    process_fx_pair(Currency.EUR, Currency.USD)    
    process_fx_pair(Currency.USD, Currency.AUD)
