from src.data import constants
from dataclasses import dataclass
from typing import List

import dask.dataframe as dd 
import numpy as np
import os
import logging


log = logging.getLogger("Raw Data Generator")


@dataclass
class DataPreprocessor:
    files: List[str]
    output_dir = f"{constants.ROOT_DIR}/data/raw/"
    
    def _cache_parquet_data(self, clobber: bool = False) -> str:
        """ Save the passed CSV files into Parquet files for a given currency 
        pair. 
        
        Args:
            clobber (bool): Whether overwrite data or not.
        
        Returns:
            str: The output directory-.
        """
        fx_pair = self._get_fx_pair()
        if (not clobber) & os.path.isdir(self.output_dir + "{fx_pair}/"):
            log.info(f"Data already exists in {self.output_dir}{fx_pair}/")
            return self.output_dir
        
        df = self._load_files()
        df.to_parquet(self.output_dir + f"{fx_pair}/")
        log.info(f"Data for {fx_pair} has been saved to \"{self.output_dir}\"")
        return self.output_dir
    
    def _get_fx_pair(self) -> str:
        """ Get the currency pair name from data paths.

        Returns:
            str: The currency pair name.        
        """
        file = self.files[0]
        filename = file.split("/")[-1]
        pair = filename.split("-")[0]
        log.info(f"The currency pair to save is {pair}")
        return pair
    
    def read_csv_dask(self, file_pos: int) -> dd.DataFrame:
        """ Read and process the unzipped CSV files to the desired format.

        Args:
            file (int): position of the path to the CSV file to read.

        Returns:
            dd.DataFrame: distributed formatted dataframe.
        """
        df = dd.read_csv(
            self.files[file_pos], 
            header=None, 
            usecols=[1, 2, 3], 
            names=constants.col_names
        )
        df['time'] = dd.to_datetime(df.time)
        df['spread'] = df['high'] - df['low']
        df['mid'] = (df['low'] + df['high']) / 2
        df = df.drop(['high', 'low'], axis=1)
        return df.astype({'mid': np.float32, 'spread': np.float32})

    def _load_files(self) -> dd.DataFrame:
        """ Load all files preprocessed, and returns a single DataFrame with 3
        columns containing the low, mid, and high prices, indexed by the time.
        
        Returns:
            dd.DataFrame: A DataFrame with the whole data.
        """
        dfs = [self.read_csv_dask(i) for i in range(len(self.files))]
        df = dd.concat(dfs)
        log.debug("DataFrame tasks have been defined.")
        return df
