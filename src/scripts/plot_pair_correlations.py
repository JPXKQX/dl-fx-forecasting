from src.data import constants
from src.scripts.click_utils import CurrencyType
from src.visualization.plot_correlations import PlotCorrelationHeatmap
from src.visualization.plot_correlations import PlotACFCurreny

import click
import logging


DATE_TYPE = click.DateTime()
CURRENCY_TYPE = CurrencyType()
CURRENCY_VARS = click.Choice(['mid', 'spread', 'increment'], case_sensitive=False)
AGG_FRAMES = click.Choice(['H', 'S'], case_sensitive=False)


@click.command()
@click.argument('var', type=CURRENCY_VARS)
@click.option('--agg_frame', type=AGG_FRAMES, default='H', help='Timeframe to '
              'which to aggregate the data.')
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default=f"{constants.ROOT_DIR}/data/raw/", 
              type=click.STRING, help="Path to the folfer containing the "
              "different currency pairs.", metavar="<str>")
def main_corrs(var, agg_frame, period, data_path):
    """ Plot the heatmap of the correlation between the different currency 
    pairs.

    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    
    PlotCorrelationHeatmap(var, data_path, agg_frame).plot_heatmap(period)


@click.command()
@click.argument('var', type=CURRENCY_VARS)
@click.argument('base_currency', type=CURRENCY_TYPE)
@click.argument('ref_currency', type=CURRENCY_TYPE)
@click.option('--agg_frame', type=AGG_FRAMES, default='H', help='Timeframe to '
              'which to aggregate the data.')
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default=f"{constants.ROOT_DIR}/data/raw/", 
              type=click.STRING, help="Path to the folfer containing the "
              "different currency pairs.", metavar="<str>")
def main_acf(var, base_currency, ref_currency, agg_frame, period, data_path):
    """ Plot the cross correlation for all currency pairs with the base currency
    and comparing to the pair with the base currency and the reference currency.

    VAR is the variable to represent. Choices are \'mid\', \'spread' and
    \'increment\'. \n
    BASE_CURRENCY is the currency contained in all pairs to compare. \n
    REF_CURRENCY is the other currency to take as base currency pair. \n
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    
    PlotACFCurreny(base_currency, var, agg_frame, path=data_path).run(ref_currency, period)
