from src.visualization.plot_trace_examples import PlotCurrencyPair
from src.data.constants import Currency


def test_line_plot():
    PlotCurrencyPair(
        Currency.EUR,
        Currency.USD,
        ['D', 'H']
    ).run(('2020-04-01', '2020-06-01'))