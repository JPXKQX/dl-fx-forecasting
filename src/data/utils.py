from typing import List, Union
from src.data.constants import col_names, ROOT_DIR, timezone
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import os
import glob
import zipfile
import logging


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


def list_all_fx_pairs(path: str = f"{ROOT_DIR}data/raw/") -> List[str]:
    """ Get all the currency pairs downloaded.

    Args:
        path (str): Path to the raw data folder. Defaults to WORKING_DIR + 
        "data/raw/".

    Returns:
        List[str]: Currency pairs.
    """
    fx_pairs = next(os.walk(path))[1]
    return fx_pairs