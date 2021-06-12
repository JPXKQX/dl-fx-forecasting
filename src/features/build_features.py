import pandas as pd
import numpy as np


def get_xy(data: pd.DataFrame, past_ticks: int, ticks_ahead: int) -> np.ndarray:
    x_inc = pd.DataFrame(columns=list(range(past_ticks)))
    x_spread = pd.DataFrame(columns=list(range(past_ticks)))

    global row
    row = 0

    def compute(window, df):
        global row
        df.loc[row, :] = window.values
        row += 1    
        return 1
    
    data.increment.rolling(past_ticks).apply(compute, kwargs={'df': x_inc})
    row = 0
    data.spread.rolling(past_ticks).apply(compute, kwargs={'df': x_spread})
    x = pd.concat([x_inc, x_spread], axis=1)
    y = data.loc[past_ticks + ticks_ahead - 1:].increment
    return x, y
    