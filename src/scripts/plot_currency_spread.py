from src.visualization.currency_spread import PlotCDFCurrencySpread, PlotStatsCurrencySpread
from src.scripts.click_utils import CurrencyType, AggregationType

import click
import logging

AGGREGATE_FRAME = AggregationType()
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
def main_cdf(base, quote, period, data_path, rate):
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
    
    PlotCDFCurrencySpread(base, quote, rate, data_path).run(period)


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.argument('agg_frame', type=AGGREGATE_FRAME)
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default="data/raw/", type=click.STRING, help="Path"
              " to the folfer containing the different currency pairs.", 
              metavar="<str>")
@click.option('--rate', default=1e3, type=click.INT, metavar='<int>', 
              help="Rate to augment the spread of the currency pair.")
@click.option('--include_max', default=False, type=click.BOOL,
              metavar='<boolean>', help='Whether to include the max spread.')
def main_stats(base, quote, agg_frame, period, data_path, rate, include_max):
    """ Show the boxplot of the main statistics of the spread of a currency 
    pair. 

    BASE is the base  to plot. For example: pass \'eur\' (case insensitive) to 
    consider euros. \n
    QUOTE is the quote currency to plot. \n
    AGG_FRAME represents the aggregation timeframe of the spread statistics. The
    possible options are: \'M\' (for monthly), \'W\' (for weekly), or \'D\' (for
     daily).
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    
    pscs = PlotStatsCurrencySpread(base, quote, agg_frame, rate, data_path)
    pscs.run(period, include_max)
