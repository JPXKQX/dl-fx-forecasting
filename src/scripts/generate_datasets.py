from src.data.data_extract import DataExtractor
from src.data.data_preprocess import DataPreprocessor
from src.data.constants import Currency
from typing import NoReturn

import logging
import click


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s | %(levelname)s | %(message)s')


class CurrencyType(click.ParamType):
    def convert(self, value, param, ctx):
        if isinstance(value, str):
            try:
                return Currency(value).name
            except ValueError:
                self.fail(f"{value!r} is not a valid string", param, ctx)

CURRENCY_TYPE = CurrencyType()

@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.option('--clobber', defautl=False, type=click.BOOL,
              help="If data should be re-generated or not")
def process_fx_pair(
    base: Currency, 
    quote: Currency, 
    clobber: bool = False
) -> NoReturn:
    """ Process the data corresponding to currency pair specified.

    Args:
        base (Currency): base currency to consider.
        qoute (Currency): quote currency to consider.
        clobber (bool): if true overwrite the data associated.
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


if __name__ == '__main__':
    process_fx_pair(Currency.AUD, Currency.JPY)
    process_fx_pair(Currency.AUD, Currency.JPY)
    process_fx_pair(Currency.AUD, Currency.USD)
    process_fx_pair(Currency.CAD, Currency.JPY)
    process_fx_pair(Currency.CHF, Currency.JPY)
    process_fx_pair(Currency.EUR, Currency.CHF)
    process_fx_pair(Currency.EUR, Currency.GBP)
    process_fx_pair(Currency.EUR, Currency.JPY)
    process_fx_pair(Currency.EUR, Currency.PLN)
    process_fx_pair(Currency.EUR, Currency.USD)
    process_fx_pair(Currency.GBP, Currency.JPY)
    process_fx_pair(Currency.GBP, Currency.USD)
    process_fx_pair(Currency.NZD, Currency.USD)
    process_fx_pair(Currency.USD, Currency.CAD)
    process_fx_pair(Currency.USD, Currency.CHF)
    process_fx_pair(Currency.USD, Currency.JPY)
    process_fx_pair(Currency.USD, Currency.MXN)
    process_fx_pair(Currency.USD, Currency.RUB)
    process_fx_pair(Currency.USD, Currency.TRY)
    process_fx_pair(Currency.USD, Currency.ZAR)