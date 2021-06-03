from click.testing import CliRunner
from pytest_mock import MockerFixture
from src.visualization import line_plot, currency_spread, plot_hourly_correlation
from src.scripts.plot_currency_pair import main
from src.data.constants import Currency

import pytest
import pandas as pd
import numpy as np


def mock_data(args):
    df = pd.read_csv("tests/data/EURUSD-parquet.csv")
    df = df.set_index(pd.to_datetime(df['time']))
    df = df.drop('time', axis=1)
    return df.add(np.random.normal(0, 1, df.shape))


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
    currency_spread.PlotCDFCurrencySpread(
        Currency.EUR,
        Currency.USD,
        1000
    ).run()
    

def test_spread_boxplot(mocker: MockerFixture):
    mocker.patch.object(
        currency_spread.DataLoader, 
        'read',
        return_value=mock_data()
    )
    currency_spread.PlotStatsCurrencySpread(
        Currency.EUR,
        Currency.USD,
        'per second'
    ).run()
    

def test_heatmap_corrs(mocker: MockerFixture):
    mocker.patch.object(
        currency_spread.DataLoader, 
        'read',
        side_effect=mock_data
    )
    plot_hourly_correlation.PlotCorrelationHeatmap(
        'mid',
        "data/raw/", 
        's'
    ).plot_heatmap()
    

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