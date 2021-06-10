from click.testing import CliRunner
from pytest_mock import MockerFixture
from tests.mocks import *
from src.data import constants
from src.visualization import line_plot, currency_pair, plot_correlations
from src.scripts import plot_currency_pair, plot_pair_correlations

import pytest


def test_cli_line_plot(mocker: MockerFixture):
    mocker.patch.object(
        line_plot.DataLoader,
        'read',
        return_value=mock_data()
    )
    runner = CliRunner()
    result = runner.invoke(
        plot_currency_pair.main, ['eur', 'usd', 'T', 'none'])
    
    assert result.exit_code == 0
    assert 'Data is prepared to be shown.' in result.output
    assert '| Line plotter | INFO |' in result.output


def test_cli_spread_plots(mocker: MockerFixture):
    mocker.patch.object(
        line_plot.DataLoader,
        'read',
        return_value=mock_data()
    )
    runner = CliRunner()
    result1 = runner.invoke(
        plot_currency_pair.main_stats, ['eur', 'usd', 'increment', 's'])
    result2 = runner.invoke(
        plot_currency_pair.main_cdf, ['eur', 'usd', 'spread'])
    result3 = runner.invoke(
        plot_currency_pair.main_cdf, ['eur', 'usd', 'spred'])

    assert result1.exit_code == 0
    assert result2.exit_code == 0
    assert result3.exit_code == 2


def test_cli_heatmap(mocker: MockerFixture):
    mocker.patch.object(
        currency_pair.DataLoader, 
        'read',
        side_effect=mock_data_randomly
    )
    mocker.patch.object(
        plot_correlations.utils, 
        'list_all_fx_pairs',
        return_value=['EUR/PLN', 'USD/MXN', 'NZD/USD', 'USD/TRY', 'EUR/JPY', 
                      'AUD/USD', 'CHF/JPY', 'CAD/JPY', 'USD/ZAR', 'EUR/USD', 
                      'AUD/JPY', 'GBP/JPY', 'USD/CAD', 'EUR/CHF', 'USD/CHF', 
                      'EUR/GBP', 'USD/JPY', 'GBP/USD']
    )
    runner = CliRunner()
    result = runner.invoke(
        plot_pair_correlations.main, ['mid', '--agg_frame', 'S'])
    
    assert result.exit_code == 0
