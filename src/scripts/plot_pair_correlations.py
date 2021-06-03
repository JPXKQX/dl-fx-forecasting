from src.data import constants
from src.visualization.plot_hourly_correlation import PlotCorrelationHeatmap

import click
import logging


DATE_TYPE = click.DateTime()
CURRENCY_VARS = click.Choice(['mid', 'spread'], case_sensitive=False)
AGG_FRAMES = click.Choice(['H', 'S'], case_sensitive=False)


@click.command()
@click.argument('var', type=CURRENCY_VARS)
@click.option('--agg_frame', type=AGG_FRAMES, default='H', help='Timeframe to '
              'which to aggregate the data.')
@click.option('--period', type=click.Tuple([DATE_TYPE, DATE_TYPE]), 
              default=None, help="Period of time to plot.")
@click.option('--data_path', default=f"{constants.ROOT_DIR}/data/raw/", type=click.STRING, help="Path"
              " to the folfer containing the different currency pairs.", 
              metavar="<str>")
def main(var, agg_frame, period, data_path):
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
