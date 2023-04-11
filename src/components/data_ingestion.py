import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import read_raw_data, write_raw_data
from src.exception import CustomException
from cfg.config import Config
from cfg.paths import Paths
import sys


class DataIngestion:
    def __init__(self):
        self.config_obj = Config()
        self.paths_obj = Paths()

    def split_data(self, df, test_size, random_state):
        """
        Splits the dataframe into training and test sets.
        """
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def ingest_data(self):
        """
        Ingests the raw data, splits it into training and test sets,
        and saves the resulting dataframes to CSV files.

        Parameters:
        -----------
        config: dict
            The configuration dictionary containing the necessary parameters for ingestion.
        paths: Namespace
            The namespace containing the necessary file paths for ingestion.

        Returns:
        --------
        Tuple[str, str]
            A tuple of file paths to the training and test data CSV files.
        """
        try:
            logging.info("Entered the data ingestion method or component")

            raw_data_dir = self.paths_obj.data_ingestion.raw_data_dir
            raw_data_file = self.paths_obj.data_ingestion.raw_data_file
            raw_data_path  = Path("..") / ".." / raw_data_dir / raw_data_file
            logging.info(f"Raw data file path: {raw_data_path}")

            df = read_raw_data(raw_data_path)

            logging.info('Read the dataset as dataframe')

            artifacts_dir = self.paths_obj.artifacts_dir
            raw_data_cpy = self.paths_obj.data_ingestion.raw_data_cpy
            raw_data_copy_path = Path("..") / ".." / artifacts_dir / raw_data_cpy

            write_raw_data(df, raw_data_copy_path)

            logging.info("Train test split initiated")

            train_set, test_set = self.split_data(
                df,
                test_size=self.config_obj.data_ingestion.ttsplit_test_size,
                random_state=self.config_obj.data_ingestion.ttsplit_random_state
            )

            train_data_file = self.paths_obj.data_ingestion.train_data_file
            train_data_path = Path("..") / ".." / artifacts_dir / train_data_file
            write_raw_data(train_set, train_data_path)

            test_data_file = self.paths_obj.data_ingestion.test_data_file
            test_data_path = Path("..") / ".." / artifacts_dir / test_data_file
            write_raw_data(test_set, test_data_path)

            logging.info("Ingestion of the data is completed")

            return train_data_path, test_data_path

        except Exception as e:
            logging.exception(e)
            raise CustomException(f"Error in DataIngestion.ingest_data: {e}", sys)
