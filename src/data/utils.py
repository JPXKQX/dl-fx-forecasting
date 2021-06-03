from typing import List, Union, Tuple
from src.data.constants import ROOT_DIR, timezone
from datetime import datetime

import os
import glob
import zipfile
import logging
import pandas as pd


log = logging.getLogger("Data Preprocessor")

def unzip_files(paths: List[str]) -> List[str]:
    """ Unzips all ZIP files whose paths are passed as arguments.

    Args:
        paths (List[str]): paths to the ZIP files to unzip.
        
    Returns:
        List[str]: paths to the unzipped folder.
    """
    unzipped_folders = []
    
    for path in paths:
        unzipped_folders.append(path.replace('.zip', ''))
        
        if not os.path.isdir(unzipped_folders[-1]):
            log.debug(f"Extracting data to {path} ...")
            with zipfile.ZipFile(path, 'r') as zipped:
                zipped.extractall(unzipped_folders[-1])
        else:
            log.debug(f"Folder {unzipped_folders[-1]} is already uncompressed.")     
    
    return unzipped_folders    


def get_files_of_extension(folders: List[str], extension: str) -> List[str]: 
    """ Get the files in the with specified extension in the list of folders 
    specified by the arguments.

    Args:
        folders (List[str]): folder in which to look for the files.
        extension (str): extension of the files to look for.

    Returns:
        List[str]: filenames found
    """
    files = []
    
    for folder in folders:
        files.extend(glob.glob(f"{folder}/*.{extension}"))
        
    return files


def str2datetime(date: Union[str, datetime]) -> datetime:
    """ Convert from string to datetime.

    Args:
        string (Union[str, datetime.datetime]): string representing a date.

    Returns:
        datetime: date object
    """
    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")
    
    return timezone.localize(date)


def period2str(
    date_interval: Union[Tuple[Union[str, datetime], 
                               Union[str, datetime]],
                         None]
    ) -> str:
    """ Generate a string of period of time passed as parameter.

    Args:
        date_interval: Period of time specified either by strings or datetimes.

    Returns:
        str: String of period of time covered.
    """
    if date_interval is None: return ""
    conv_func = lambda x: str2datetime(x).strftime("%d %b %Y")
    date_strs = list(map(conv_func, date_interval))
    return f" (From {date_strs[0]} to {date_strs[1]})"
                

def filter_datetime_series(
    series: pd.Series, 
    agg_timeframe: str
    ) -> Tuple[pd.Series, str]: 
    """ Convert datetime series to the timeframe specified. 

    Args:
        series (dd.Series): datetime series
        agg_timeframe (str): aggregation timeframe

    Raises:
        NotImplementedError: the timeframe specified is not supported.

    Returns:
        dd.Series: the resulting series.
    """
    if agg_timeframe.lower() in ['s', 'per second']:
        return series.floor('s').second, 'Per second'
    elif agg_timeframe.lower() in ['d', 'daily']:
        return series.floor('d'), 'Daily'
    elif agg_timeframe.lower() in ['w', 'weekly']:
        return series.weekofyear, 'Weekly'
    elif agg_timeframe.lower() in ['m', 'monthly']:
        return series.astype('<M8[h]'), 'Monthly'
    else:
        raise NotImplementedError(f"Aggregate timeframe not implemented.")


def list_all_fx_pairs(path: str = f"{ROOT_DIR}/data/raw/") -> List[str]:
    """ Get all the currency pairs downloaded.

    Args:
        path (str): Path to the raw data folder. Defaults to WORKING_DIR + 
        "data/raw/".

    Returns:
        List[str]: Currency pairs.
    """
    fx_pairs = next(os.walk(path))[1]
    return [f"{pair[:3]}/{pair[3:]}" for pair in fx_pairs]