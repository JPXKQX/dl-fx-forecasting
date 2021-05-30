from click.testing import CliRunner
from pytest_mock import MockerFixture
from src.visualization.line_plot import PlotCurrencyPair, DataLoader
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
        DataLoader, 
        'read',
        return_value=mock_data()
    )
    PlotCurrencyPair(
        Currency.EUR,
        Currency.USD,
        ['S', None]
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