from src.data.data_extract import DataExtractor
from src.data.data_preprocess import DataPreprocessor
from src.data.constants import Currency
from typing import NoReturn
from src.scripts.click_utils import CurrencyType

import logging
import click


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s | %(levelname)s | %(message)s')

CURRENCY_TYPE = CurrencyType()


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.option('--clobber', default=False, type=click.BOOL, 
              help="If data should be re-generated or not")
def process_fx_pair(
    base: Currency, 
    quote: Currency, 
    clobber: bool = False
) -> NoReturn:
    """ Process the data corresponding to currency pair BASE/QUOTE specified.

    BASE is base currency to consider.\n
    QUOTE is the quote currency to consider.\n
    """   
    # Unzip data
    dd_2020 = DataExtractor((base, quote), list(range(4, 12)), 2020)
    csv_files_2020 = dd_2020.prepare()
    dd_2021 = DataExtractor((base, quote), [1, 2, 3, 4], 2021)
    csv_files_2021 = dd_2021.prepare() 
    csv_files = csv_files_2020 + csv_files_2021
    
    # Save into Parquet files.
    dp = DataPreprocessor(csv_files)
    dp._cache_parquet_data(clobber) 
