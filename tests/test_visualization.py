from click.testing import CliRunner
from pytest_mock import MockerFixture
from src.visualization.line_plot import PlotCurrencyPair
from src.scripts.plot_currency_pair import main
from src.data.constants import Currency

import pytest


def test_line_plot(mocker: MockerFixture):
    mocker.patch.object(
        PlotCurrencyPair, 
        '_search_pair',
        return=("tests/data/EURUSD-parquet.csv", False)
    )
    PlotCurrencyPair(
        Currency.EUR,
        Currency.USD,
        ['M', 'none']
    ).run(('2020-05-01', '2021-05-31'))
    

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