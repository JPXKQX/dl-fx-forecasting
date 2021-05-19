from src.data import utils
from dataclasses import dataclass
from typing import List

import dask.dataframe as dd 
import os
import logging


log = logging.getLogger("Dataset Generator")
WORKING_DIR = os.environ['PWD']


@dataclass
class DataPreprocessor:
    files: List[str]
    
    def save_datasets(self, clobber: bool = False) -> str:
        """ Save the passed CSV files into Parquet files for a given currency 
        pair. 
        
        Args:
            clobber (bool): Whether overwrite data or not.
        
        Returns:
            str: The output directory-.
        """
        fx_pair = self._get_fx_pair()
        output_dir = f"{WORKING_DIR}/data/raw/{fx_pair}/"
        if (not clobber) & os.path.isdir(output_dir):
            log.info(f"Data already exists in {output_dir}")
            return output_dir
        
        df = self._load_files()
        df.to_parquet(output_dir)
        log.info(f"Data for {fx_pair} has been saved to \"{output_dir}\"")
        return output_dir
    
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
    
    def _load_files(self) -> dd.DataFrame:
        """ Load all files preprocessed, and returns a single DataFrame with 3
        columns containing the low, mid, and high prices, indexed by the time.
        
        Returns:
            dd.DataFrame: A DataFrame with the whole data.
        """
        dfs = [utils.read_csv_dask(f) for f in self.files]
        df = dd.concat(dfs)
        df['mid'] = (df['low'] + df['high']) / 2
        log.debug("DataFrame tasks have been defined.")
        return df
