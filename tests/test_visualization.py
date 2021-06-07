from tests.mocks import *
from pytest_mock import MockerFixture
from src.visualization import line_plot, currency_spread, plot_hourly_correlation
from src.data.constants import Currency, ROOT_DIR

import pytest


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
    currency_spread.PlotCDFCurrencyPair(
        Currency.EUR,
        Currency.USD,
        'spread',
        1000
    ).run()
    

def test_spread_boxplot(mocker: MockerFixture):
    mocker.patch.object(
        currency_spread.DataLoader, 
        'read',
        return_value=mock_data()
    )
    currency_spread.PlotStatsCurrencyPair(
        Currency.EUR,
        Currency.USD,
        'spread',
        'per second'
    ).run()
    

def test_heatmap_corrs(mocker: MockerFixture):
    mocker.patch.object(
        currency_spread.DataLoader, 
        'read',
        side_effect=mock_data_randomly
    )
    mocker.patch.object(
        plot_hourly_correlation.utils, 
        'list_all_fx_pairs',
        return_value=['EUR/PLN', 'USD/MXN', 'NZD/USD', 'USD/TRY', 'EUR/JPY', 
                      'AUD/USD', 'CHF/JPY', 'CAD/JPY', 'USD/ZAR', 'EUR/USD', 
                      'AUD/JPY', 'GBP/JPY', 'USD/CAD', 'EUR/CHF', 'USD/CHF', 
                      'EUR/GBP', 'USD/JPY', 'GBP/USD']
    )
    plot_hourly_correlation.PlotCorrelationHeatmap(
        'mid',
        f"{ROOT_DIR}/data/raw/", 
        's'
    ).plot_heatmap()
