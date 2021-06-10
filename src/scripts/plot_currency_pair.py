from src.visualization.line_plot import PlotCurrencyPair
from src.visualization.currency_pair import PlotCDFCurrencyPair
from src.visualization.currency_pair import PlotStatsCurrencyPair
from src.scripts.click_utils import CurrencyType, AggregationType

import click
import logging

CURRENCY_TYPE = CurrencyType()
AGG_FRAME = AggregationType()
DATE_TYPE = click.DateTime()


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.argument('freqs', nargs=-1, type=AGG_FRAME)
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


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.argument('which', type=click.Choice(['spread', 'increment']))
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default="data/raw/", type=click.STRING, help="Path"
              " to the folfer containing the different currency pairs.", 
              metavar="<str>")
@click.option('--rate', default=1e3, type=click.INT, metavar='<int>', 
              help="Rate to augment the spread of the currency pair.")
def main_cdf(base, quote, which, period, data_path, rate):
    """ Plot the Cumulative Distribution function (CDF) of the currency pair
    chosen.

    BASE is the base  to plot. For example: pass \'eur\' (case insensitive) to 
    consider euros. \n
    QUOTE is the quote currency to plot. \n
    WHICH indicates the feature to show. \n
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    
    PlotCDFCurrencyPair(base, quote, which, rate, data_path).run(period)


@click.command()
@click.argument('base', type=CURRENCY_TYPE)
@click.argument('quote', type=CURRENCY_TYPE)
@click.argument('which', type=click.Choice(['spread', 'increment']))
@click.argument('agg_frame', type=AGG_FRAME)
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default="data/raw/", type=click.STRING, help="Path"
              " to the folfer containing the different currency pairs.", 
              metavar="<str>")
@click.option('--rate', default=1e3, type=click.INT, metavar='<int>', 
              help="Rate to augment the spread of the currency pair.")
@click.option('--include_max', default=False, type=click.BOOL,
              metavar='<boolean>', help='Whether to include the max spread.')
def main_stats(base, quote, which, agg_frame, period, data_path, rate, include_max):
    """ Show the boxplot of the main statistics of the spread of a currency 
    pair. 

    BASE is the base  to plot. For example: pass \'eur\' (case insensitive) to 
    consider euros. \n
    QUOTE is the quote currency to plot. \n
    WHICH indicates the feature to show. \n
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
    
    pscs = PlotStatsCurrencyPair(base, quote, which, agg_frame, rate, data_path)
    pscs.run(period, include_max)
