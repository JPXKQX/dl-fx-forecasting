import pandas as pd
import numpy as np


def mock_data():
    df = pd.read_csv("tests/data/EURUSD-parquet.csv")
    df = df.set_index(pd.to_datetime(df['time']))
    df['increment'] = df.mid.diff()
    df = df.drop('time', axis=1)
    return df


def mock_data_randomly(args):
    df = pd.read_csv("tests/data/EURUSD-parquet.csv")
    df = df.set_index(pd.to_datetime(df['time']))
    df['increment'] = df.mid.diff()
    df = df.drop('time', axis=1)
    return df.add(np.random.normal(0, 1, df.shape))
