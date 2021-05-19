from enum import Enum 


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
        
col_names = ["time", "low", "high"]
