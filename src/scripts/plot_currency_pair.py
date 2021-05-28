from src.visualization.line_plot import PlotCurrencyPair
from src.data.constants import Currency

import argparse
import logging


def init_parser() -> argparse.Namespace:
    """ Build a parser for the arguments specified when producing the plots.

    Returns:
        parsed_args (NameSpace): each parameter converted to the 
        appropriate type.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-base", help="Choose the base currency to plot. "
                        "For example: pass \'eur\' (case insensitive) to "
                        "consider euros.", 
                        type=str)
    parser.add_argument("-quote", help="Choose the quote currency to plot. "
                        "For example: pass \'eur\' (case insensitive) to "
                        "consider euros.", 
                        type=str)
    parser.add_argument("--period", help="Period of time to plot.", 
                        default=None)
    parser.add_argument("--freqs", default=['D', 'H'], help="")
    parser.add_argument("--data_path", help="Password for Nextcloud.",
                        default="data/raw/")
    return parser.parse_args()


if __name__ == "__main__":
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )
    args = init_parser()
    PlotCurrencyPair(
        Currency(args.base.upper()).name,
        Currency(args.quote.upper()).name,
        args.freqs,
        args.data_path
    ).run(args.period)