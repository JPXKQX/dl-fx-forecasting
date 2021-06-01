from src.visualization.currency_spread import PlotCurrencySpread
from src.scripts.click_utils import CurrencyType

import click
import logging

CURRENCY_TYPE = CurrencyType()
DATE_TYPE = click.DateTime()


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default="data/raw/", type=click.STRING, help="Path"
              " to the folfer containing the different currency pairs.", 
              metavar="<str>")
@click.option('--rate', default=1e3, type=click.INT, metavar='<int>', 
              help="Rate to augment the spread of the currency pair.")
def main(base, quote, period, data_path, rate):
    """ Plot the Cumulative Distribution function (CDF) of the currency pair
    chosen.

    BASE is the base  to plot. For example: pass \'eur\' (case insensitive) to 
    consider euros. \n
    QUOTE is the quote currency to plot. \n
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    
    PlotCurrencySpread(base, quote, rate, data_path).run(period)
