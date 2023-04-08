import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataIngestion:
	def __init__(self, config, paths):
		self.config = config
		self.paths = paths

	def initiate_data_ingestion(self):
		try:
			logging.info("Entered the data ingestion method or component")
			root = str(Path(os.getcwd()).parents[1])

			raw_data_dir = self.paths['data_ingestion']['raw_data_dir']
			raw_data_file = self.paths['data_ingestion']['raw_data_file']
			file_path = os.path.join(root, raw_data_dir, raw_data_file )

			df = pd.read_csv(file_path)

			logging.info('Read the dataset as dataframe')
			print("Read the dataset as dataframe")

			artifacts_dir = self.paths['artifacts_dir']
			raw_data_cpy = self.paths['data_ingestion']['raw_data_cpy']
			raw_data_path = os.path.join(root, artifacts_dir, raw_data_cpy)

			os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

			df.to_csv(raw_data_path, index=False, header=True)

			logging.info("Train test split initiated")

			train_set, test_set = \
				train_test_split(df, test_size = self.config['data_ingestion']['ttsplit_test_size'],
			                        random_state = self.config['data_ingestion']['ttsplit_random_state'])

			train_data_file = self.paths['data_ingestion']['train_data_file']
			test_data_file = self.paths['data_ingestion']['test_data_file']
			train_data_path = os.path.join(root, artifacts_dir, train_data_file)
			test_data_path = os.path.join(root, artifacts_dir, test_data_file)

			train_set.to_csv(train_data_path, index=False, header=True)
			test_set.to_csv(test_data_path, index=False, header=True)

			logging.info("Ingestion of the data is completed")

			return (
				train_data_path,
				test_data_path
			)

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)




