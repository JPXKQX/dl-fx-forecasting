from enum import Enum 

import os


WORKING_DIR = os.environ['PWD']

months = ["None",
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December"
]


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
