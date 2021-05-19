from typing import List
from src.data.constants import col_names

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


def read_csv_dask(file: str) -> dd.DataFrame:
    """ Read the unzipped CSV files to the desired format.

    Args:
        file (str): path to the CSV file.

    Returns:
        dd.DataFrame: distributed formatted dataframe.
    """
    return dd.read_csv(
        file, 
        header=None, 
        usecols=[1, 2, 3], 
        parse_dates=[1], 
        infer_datetime_format=True,
        dtype={'low': np.float32, 'high': np.float32},
        names=col_names
    ).set_index('time')