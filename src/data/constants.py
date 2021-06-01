from enum import Enum 

import os
import pytz

ROOT_DIR = os.path.dirname(os.path.abspath("setup.py"))
PATH_RAW_DATA = os.getenv('PATH_RAW_DATA')

months = ["None",
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December"
]

stat2label = {
    'min': 'Minimum',
    'mean': 'Mean',
    'median': 'Median',
    'std': 'Standard Deviation',
    'max': 'Maximum',
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
