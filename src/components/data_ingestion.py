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

            root = Path(__file__).parent.parent.parent.resolve()
            logging.info(f"Project root: {root}")

            raw_data_dir = self.paths_obj.data_ingestion.raw_data_dir
            raw_data_file = self.paths_obj.data_ingestion.raw_data_file
            raw_data_path = root / raw_data_dir / raw_data_file
            logging.info(f"Raw data file path: {raw_data_path}")

            df = read_raw_data(raw_data_path)

            logging.info('Read the dataset as dataframe')

            artifacts_dir = self.paths_obj.artifacts_dir
            raw_data_cpy = self.paths_obj.data_ingestion.raw_data_cpy
            raw_data_copy_path = root / artifacts_dir / raw_data_cpy

            write_raw_data(df, raw_data_copy_path)

            logging.info("Train test split initiated")

            train_set, test_set = self.split_data(
                df,
                test_size=self.config_obj.data_ingestion.ttsplit_test_size,
                random_state=self.config_obj.data_ingestion.ttsplit_random_state
            )

            train_data_file = self.paths_obj.data_ingestion.train_data_file
            train_data_path = root / artifacts_dir / train_data_file
            write_raw_data(train_set, train_data_path)

            test_data_file = self.paths_obj.data_ingestion.test_data_file
            test_data_path = root / artifacts_dir / test_data_file
            write_raw_data(test_set, test_data_path)

            logging.info("Ingestion of the data is completed")

            return (
                str(train_data_path.relative_to(root)),
                str(test_data_path.relative_to(root))
            )

        except Exception as e:
            logging.exception(e)
            raise CustomException(f"Error in DataIngestion.ingest_data: {e}", sys)

# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# import pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
#
#
# class DataIngestion:
# 	def __init__(self):
# 		pass
#
# 	def initiate_data_ingestion(self, config, paths):
# 		try:
# 			logging.info("Entered the data ingestion method or component")
# 			# root = str(Path(os.getcwd()).parents[1])
#
# 			root = os.path.dirname(os.path.abspath(__file__))
# 			print(root)
# 			raw_data_dir = paths['data_ingestion']['raw_data_dir']
# 			raw_data_file = paths['data_ingestion']['raw_data_file']
# 			file_path = os.path.join(root, raw_data_dir, raw_data_file )
#
# 			df = pd.read_csv(file_path)
#
# 			logging.info('Read the dataset as dataframe')
# 			print("Read the dataset as dataframe")
#
# 			artifacts_dir = paths['artifacts_dir']
# 			raw_data_cpy = paths['data_ingestion']['raw_data_cpy']
# 			raw_data_path = os.path.join(root, artifacts_dir, raw_data_cpy)
#
# 			os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
#
# 			df.to_csv(raw_data_path, index=False, header=True)
#
# 			logging.info("Train test split initiated")
#
# 			train_set, test_set = \
# 				train_test_split(df, test_size = config['data_ingestion']['ttsplit_test_size'],
# 			                        random_state = config['data_ingestion']['ttsplit_random_state'])
#
# 			train_data_file = paths['data_ingestion']['train_data_file']
# 			test_data_file = paths['data_ingestion']['test_data_file']
# 			train_data_path = os.path.join(root, artifacts_dir, train_data_file)
# 			test_data_path = os.path.join(root, artifacts_dir, test_data_file)
#
# 			train_set.to_csv(train_data_path, index=False, header=True)
# 			test_set.to_csv(test_data_path, index=False, header=True)
#
# 			logging.info("Ingestion of the data is completed")
#
# 			return (
# 				train_data_path,
# 				test_data_path
# 			)
#
# 		except Exception as e:
# 			logging.exception(e)
# 			raise CustomException(e, sys)
#
#
#

