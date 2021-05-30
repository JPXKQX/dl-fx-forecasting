from click.testing import CliRunner
from src.visualization.line_plot import PlotCurrencyPair
from src.scripts.plot_currency_pair import main
from src.data.constants import Currency

import pytest


@pytest.mark.skip("Run with raw data processed.")  
def test_line_plot():
    PlotCurrencyPair(
        Currency.EUR,
        Currency.USD,
        ['D', 'H']
    ).run(('2020-05-01', '2020-05-31'))
    

@pytest.mark.skip("Run with raw data processed.")  
def test_cli_line_plot():
    runner = CliRunner()
    result = runner.invoke(
        main,
        ['eur', 'usd', 'T', 'none', '--period', '2020-05-24', '2020-05-26']
    )
    
    assert result.exit_code == 0
    assert 'Data is prepared to be shown.' in result.output
    assert '| Line plotter | INFO |' in result.output