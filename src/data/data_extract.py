from dataclasses import dataclass
from typing import NoReturn, Tuple, List
from src.data import utils, constants
from src.data.constants import Currency

import os
import glob
import logging

log = logging.getLogger("Data Preprocessor")
path_raw_data = os.getenv('RAW_TRUEFX_PATH')


@dataclass
class DataExtractor():
    currencies: Tuple[Currency, Currency]
    months: List[int]
    year: int
    
    def prepare(self) -> NoReturn:
        zipped_months = self.get_zipped_months()
        unzipped_months = utils.unzip_files(zipped_months)
        zipped_pairs = self.get_zipped_currency_pair(unzipped_months)
        unzipped_pairs = utils.unzip_files(zipped_pairs)
        csv_files = utils.get_files_of_extension(unzipped_pairs, 'csv')
        return csv_files
        
    def get_zipped_months(self) -> List[str]:
        """ Get path to the zipped file corresponding to the month and year.

        Raises:
            FileNotFoundError: There are no zipped files for some of the periods
            indicated.

        Returns:
            List[str]: paths to the zipped files.
        """
        m_names = [constants.months[month] for month in sorted(self.months)]
        paths = [f"{path_raw_data}/{self.year}/{name}.zip" for name in m_names]
        
        # Check all paths exists
        if any([not os.path.exists(path) for path in paths]):
            raise FileNotFoundError("Some of the files does not exist.")
        
        log.debug(f"All {len(paths)} exist.")
        return paths       
        
    def get_zipped_currency_pair(self, folders: List[str]) -> List[str]:
        """ Get path to the zipped currency pait corresponding to the month and
        year specified.
 
        Args:
            folders (List[str]): unzipped folders with all currency pairs.

        Returns:
            List[str]: zipped CSV filenames of the currency pair.
        """
        names = [currency.value for currency in self.currencies]
        pairs = [''.join(names), ''.join(reversed(names))]                 
        zipped_pairs = []

        # Look for unzipped files of the currency pair specified.
        for folder in folders:
            for pair in pairs:
                paths = glob.glob(f"{folder}/{pair}*.zip")
                zipped_pairs.extend(paths)

        return zipped_pairs
