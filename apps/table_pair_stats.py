from src.visualization.currency_pair import PlotStatsCurrencyPair
from src.data import utils
from src.data.constants import Currency, ROOT_DIR
from typing import Tuple, List, Dict
from dash.dependencies import Input, Output

import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd


logger = logging.getLogger("Dash app Table")

# Variables
rate = 1
data_path = f"{ROOT_DIR}/data/raw/"
period = '2020-04-01', '2020-06-01'

# Attributes
currencies = ["EUR", "USD", "CAD", "GBP", "AUD", "JPY", 
              "CHF", "PLN", "NZD", "MXN", "TRY", "ZAR"]
statistic_names = [
    "Hour", "Std Dev", "Minimum", "0.05 Percentile", "5 Percentile", 
    "1st Quartile", "Median", "Mean", "3rd Quartile", "95 Percentile", 
    "99.95 Percentile", "Maximum"]


# Create Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

server = app.server

app.layout = html.Div([
    html.H1(
        id='title', 
        style=dict(textAlign='center')),
    html.H3(
        id='subtitle',
        style=dict(textAlign='center')),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='base-selector',
                options=list(
                    map(lambda x: dict(label=x, value=x), currencies)
                ), 
                value="EUR",
                clearable=False
            )
            ], style=dict(width='30%', display='inline-block'))
        , html.Div( [
            dcc.Dropdown(
                id='quote-selector',
                placeholder="Select another currency ..."
            )
            ], style=dict(width='30%', display='inline-block'))
        , html.Div([
            dcc.Dropdown(
                id='variable-selector',
                options=[
                    {'label': 'Price increments', 'value': 'increment'},
                    {'label': 'Mid prices', 'value': 'mid'},
                    {'label': 'Spread', 'value': 'spread'}
                ],
                clearable=False,
                value='increment'
            )
            ], style=dict(width='40%', display='inline-block'))
    ], style=dict(width='50%', margin='auto')),
    html.Div([
        dcc.Graph(id='table-variable', config=dict(displayModeBar=False))
    ], id="div-table")     
])


# Utility functions
def get_updated_figure(varname: str, base: str, quote: str):
    logger.debug(f"Updating figure for variable {varname}.")
    n_cols = len(statistic_names) - 1
    if varname is None or quote is None: 
        return go.Figure()
    filename = f"/tmp/hourly_stats_{varname}_{base}{quote}.csv"
    df = pd.read_csv(filename)
    df = df.round(4)
    fig = go.Figure(data=[go.Table(
        columnwidth= [20] + n_cols * [60],
        header=dict(values=statistic_names,
                    line_color='darkslategray',
                    fill_color='royalblue',
                    align=['left'] + n_cols * ['center'],
                    height=25,
                    font=dict(color='white', size=12)),
        cells=dict(values=df.values.T,
                    height=20,
                    line_color='darkslategray',
                    fill_color=['lightskyblue'] + n_cols * ['white'],
                    align=['left'] + n_cols * ['center'])
    )])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), 
        height=50+24*20, width=2*(40+n_cols*80))
    return fig


# Callbacks
@app.callback(
    Output('quote-selector', 'options'), Output('quote-selector', 'value'),
    Output('quote-selector', 'placeholder'), Input('base-selector', 'value')
)
def set_quote_options(base: str) -> Tuple[List, str, str]:
    other_curr = utils.list_currencies_against(Currency(base))
    options = list(map(
        lambda x: dict(label=x.value, value=x.value), 
        other_curr
    ))
    return options, None, "Select another currency ..."

@app.callback(
    Output('div-table', 'style'),
    Output('table-variable', 'figure'),
    Output('title', 'children'), 
    Output('subtitle', 'children'),
    Input('variable-selector', 'value'), 
    Input('base-selector', 'value'), 
    Input('quote-selector', 'value'),
)
def cache_data_pair(
    varname: str, 
    base: str, 
    quote: str
) -> Tuple[Dict, object, str, str]:
    logger.debug(f"Caching data for {base}/{quote}")
    if quote is None or base is None: 
        subtitl = utils.period2str(period).replace("(", "").replace(")", "")
        title = f"       Statistics Viewer"
        logger.debug(f"Setting title to: \"{title}\"")
        logger.debug(f"Setting subtitle to: \"{subtitl}\"")
        return {'display': 'none'}, go.Figure(), title, subtitl
    
    logger.info(f"{base}/{quote} stats are being computed.")
    plotter = PlotStatsCurrencyPair(
        Currency(base), Currency(quote), '', rate, data_path)
    _, subtitle, pair = plotter.get_table(period)  

    # Log results
    subtitle = subtitle.replace("(", "").replace(")", "")
    title = f"{pair} Statistics Viewer"
    logger.debug(f"Setting title to: \"{title}\"")
    logger.debug(f"Setting subtitle to: \"{subtitle}\"")
    
    fig = get_updated_figure(varname, base, quote)
    return {}, fig, title, subtitle


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False) 
