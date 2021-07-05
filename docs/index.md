<h1 align="center"> DL FX Forecasting </h1>

<h3 align="center"> Python project for forecasting changes in several Foreign Exchange (FX) pairs. </h3>

![Continuous Integration](https://github.com/JPXKQX/dl-fx-forecasting/actions/workflows/project-ci.yml/badge.svg) ![Build](https://github.com/JPXKQX/dl-fx-forecasting/actions/workflows/project-build.yml/badge.svg) ![Documentation](https://github.com/JPXKQX/dl-fx-forecasting/actions/workflows/documentation.yml/badge.svg)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<!-- ABOUT THE PROJECT -->
##  About the project :blue_book:

FX rates forecasting in ultra high frequency setting, using Deep Learning techniques. The main focus of the research is to predict the increments in the next few seconds for a set of different FX pairs.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<!-- PREREQUISITES -->
##  Prerequisites  :pushpin:


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)



<!-- ENVIRONMET -->
## Environment 

Execute the following command to start the container:

```
docker run -it --rm jpxkqx/dl-fx-forecasting:firsttry
```

In case, the data is already in processed in the host machine, the following 
command may be more appropriate.

```
docker run -it -v "/path/to/data:/app/data" --rm jpxkqx/dl-fx-forecasting:firsttry
```

_The  path **/path/to/data** refers to the directory containing the data as 
presented in the project organization below. In case all processed information is available, it is possible to execute all scripts._


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATA -->
##  Data :1234:

Read, load, preprocess and save the data for the currency pair specified. To go through this pipeline, the ZIP files have to be in the host machine, and the path to the folder containing this data must be specified as an environment variable called *PATH_RAW_DATA*. The following command process the data available in the host machine for currency pair EUR/USD.

```
generate_datasets eur usd
```

In this case, the historical data has been extracted from [True FX](https://www.truefx.com/truefx-historical-downloads/), whose first prices are shown below.


<center>

| FX pair | Timestamp | Low | High |
| -------|-----------------------|---------|---------|
|EUR/USD | 20200401 00:00:00.094 | 1.10256 | 1.10269 |
|EUR/USD | 20200401 00:00:00.105 | 1.10257 | 1.1027 |
|EUR/USD | 20200401 00:00:00.193 | 1.10258 | 1.1027 |
|EUR/USD | 20200401 00:00:00.272 | 1.10256 | 1.1027 |
|EUR/USD | 20200401 00:00:00.406 | 1.10258 | 1.1027 |
|EUR/USD | 20200401 00:00:00.415 | 1.10256 | 1.1027 |
|EUR/USD | 20200401 00:00:00.473 | 1.10257 | 1.1027 |
|EUR/USD | 20200401 00:00:00.557 | 1.10255 | 1.10268 |

</center>

This data is processed by the following command, which computes the mid price and spread and filter some erroneus data points. The processed information is stored using [Apache Parquet](https://parquet.apache.org/) in order to achieve faster reading times.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<!-- VISUALIZATION -->
## Visualizations :art:

Then, plot the currency pair *EUR/USD* for the period from 25 May, 202 to 30 May, 2020.

```
plot_currency_pair eur usd mid H T S --period 2020-05-25 2020-05-31
```

To get the following image,

![Line plot](https://github.com/JPXKQX/dl-fx-forecasting/blob/main/reports/figures/eurusd_25_30May.png?raw=True)


There is also the possibility to plot the cumulative distribution function using the following command

```
plot_cdf eur usd increment --period 2020-04-01 2020-06-01
```

which gives the image shown below,

![Cumulative Distribution Function](https://github.com/JPXKQX/dl-fx-forecasting/blob/main/reports/figures/eurusd_increments_cdf_AprMay.png?raw=True)



In order to plot the distribution of the main daily statistic of the spread, the following command can be used.

```
plot_stats eur usd spread D --period 2020-04-01 2020-06-01
```

![Boxplot](https://github.com/JPXKQX/dl-fx-forecasting/blob/main/reports/figures/eurusd_spread_dailystats_Apr_May.png?raw=True)

In addition, the correlation between the different currency pairs aggregated by any timeframe can also be plotted for any given period of time. 

```
plot_pair_correlations increment --period 2020-04-01 2020-06-01 --agg_frame H
```

![Correlation](https://github.com/JPXKQX/dl-fx-forecasting/blob/main/reports/figures/increment_correlations_AprMay.png?raw=True)


Lastly, the correlation between currency pairs is represented as follows,

```
plot_pair_acf increment eur usd --agg_frame 'H' --period 2020-04-01 2020-06-01
```

![Cross correlations](https://github.com/JPXKQX/dl-fx-forecasting/blob/main/reports/figures/increment_eur_acfs_AprMay.png?raw=True)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)







<!-- MODELLING -->
## Modelling

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)











<!-- RESULTS -->
## Results :trophy:

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)









<!-- PROJECT ORGANIZATION -->
## Project Organization :open_file_folder:


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A deafult MkDocs project.
    |   └── index.md
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |   ├── configurations <- YAML files with model configurations
    |   ├── features       <- Contains model selection results, test results and fitted models, under the path
    |   |                     models/features/{ model }/{ fx_pair }/{ aux_pair}/{ variables concat with _}
    |   |                     In particular, the models used EWMA's of a fixed number of past observations.
    │   └── raw            <- Contains model selection results, test results and fitted models, under the path
    |                         models/features/{ model }/{ fx_pair }/{ aux_pair}/{ variables concat with _}
    |                         In particular, the models used all the past observations.
    │
    ├── notebooks          <- Jupyter notebooks. Containing the results for the training process of diffferent models
    |   ├── train...hmtl   <- Output code to include in VC.
    │   └── train...ipynb  <- Python notebooks considered. Not included in VC.
    |
    |
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting, README, and docs
    │   ├── images         <- Generated graphics and figures of EDA. Not included in VC.
    │   └── models         <- Generated graphics and figures of model results. Not included in VC
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   ├── data_extract.py
    │   │   ├── data_loader.py
    │   │   ├── data_preprocess.py
    │   │   ├── utils.py
    │   │   └── constants.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── __init__.py
    │   │   ├── get_blocks.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── __init__.py
    │   │   ├── neural_network.py
    │   │   ├── model_selection.py
    │   │   ├── model_utils.py
    │   │   └── train_model.py
    │   │
    │   ├── scripts        <- Scripts to create CLI entrypoints
    │   │   ├── __init__.py
    │   │   ├── click_utils.py
    │   │   ├── generate_datasets.py
    │   │   ├── plot_currency_pair.py
    │   │   └── plot_pair_correlations.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    |       ├── __init__.py
    |       ├── line_plot.py
    |       ├── plot_correlations.py
    |       ├── plot_results.py
    │       └── currency_pair.py
    │  
    ├── tests
    │   ├── data           <- Data needed to test the functionalities.
    │   ├── mocks.py
    │   ├── test_cli_scripts.py
    │   ├── test_dataset_generation.py
    │   └── test_visualization.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- FAQ -->
## FAQ :question: 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


##References :books: 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
