from src.visualization.line_plot import PlotCurrencyPair
from src.scripts.click_utils import CurrencyType, AggregationType

import click
import logging

CURRENCY_TYPE = CurrencyType()
AGG_TYPE = AggregationType()
DATE_TYPE = click.DateTime()


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.argument('freqs', nargs=-1, type=AGG_TYPE)
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default="data/raw/", type=click.STRING, help="Path"
              " to the folfer containing the different currency pairs.", 
              metavar="<str>")
def main(base, quote, period, freqs, data_path):
    """ Plot the line plot of the currency pair BASE/QUOTE given.

    BASE is the base  to plot. For example: pass \'eur\' (case insensitive) to 
    consider euros. \n
    QUOTE is the quote currency to plot. \n
    FREQS represent the aggregation frequencies to plot. They should be listed 
    consequently. For example \'H\' represents hourly aggregated data. Other 
    options are \'D\' (for daily), \'T\' (per minute), \'S\' (per second), and 
    \'none\' for raw data.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    
    PlotCurrencyPair(base, quote, freqs, data_path).run(period)
