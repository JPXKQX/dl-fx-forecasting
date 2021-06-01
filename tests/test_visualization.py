from click.testing import CliRunner
from pytest_mock import MockerFixture
from src.visualization import line_plot, currency_spread
from src.scripts.plot_currency_pair import main
from src.data.constants import Currency

import pytest
import dask.dataframe as dd


def mock_data():
    df = dd.read_csv("tests/data/EURUSD-parquet.csv")
    df = df.set_index(dd.to_datetime(df['time']), sorted=True)
    df = df.drop('time', axis=1)
    return df


def test_line_plot(mocker: MockerFixture):
    mocker.patch.object(
        line_plot.DataLoader, 
        'read',
        return_value=mock_data()
    )
    line_plot.PlotCurrencyPair(
        Currency.EUR,
        Currency.USD,
        ['S', None]
    ).run()
    
def test_spread_plot(mocker: MockerFixture):
    mocker.patch.object(
        currency_spread.DataLoader, 
        'read',
        return_value=mock_data()
    )
    currency_spread.PlotCurrencySpread(
        Currency.EUR,
        Currency.USD,
        1000
    ).run()
    

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