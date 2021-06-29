from enum import Enum 

import os
import pytz

ROOT_DIR = os.getenv('ROOT_DIR', os.path.dirname(os.path.abspath("setup.py")))
PATH_RAW_DATA = os.getenv('PATH_RAW_DATA')

months = ["None",
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December"
]

var2label = {
    "mid": "mid prices",
    "spread": "spreads",
    "increment": "price increments"
}

stat2label = {
    'min': 'Minimum',
    'mean': 'Mean',
    'median': 'Median',
    'std': 'Standard Deviation',
    'max': 'Maximum',
    '0.05': '5th quantile',
    '0.25': 'First quartile',
    '0.5': 'Median',
    '0.75': 'Second quartile',
    '0.95': '95th quantile'
}


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    NZD = "NZD"
    CHF = "CHF"
    PLN = "PLN"
    MXN = "MXN"
    RUB = "RUB"
    TRY = "TRY"
    ZAR = "ZAR"
        
col_names = ["time", "low", "high"]

timezone = pytz.timezone('Etc/Greenwich')
