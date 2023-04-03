import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
	train_data_path: str = r'D:\projects\Insurance_Fraud_Detection\artifacts\train.csv'
	test_data_path: str = r'D:\projects\Insurance_Fraud_Detection\artifacts\test.csv'
	raw_data_path: str = r'D:\projects\Insurance_Fraud_Detection\artifacts\data.csv'
	insuranceFraud_file_path: str = r'D:\projects\Insurance_Fraud_Detection\Data\insuranceFraud.csv'
	random_state = 42
	test_size = 0.25

class DataIngestion:
	def __init__(self):
		self.ingestion_config = DataIngestionConfig()

	def initiate_data_ingestion(self):
		try:
			logging.info("Entered the data ingestion method or component")

			df = pd.read_csv(self.ingestion_config.insuranceFraud_file_path)
			logging.info('Read the dataset as dataframe')

			os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

			df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

			logging.info("Train test split initiated")

			train_set, test_set = train_test_split(df, test_size=self.ingestion_config.test_size,
			                                       random_state=self.ingestion_config.random_state)

			train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
			test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

			logging.info("Ingestion of the data is completed")

			return (
				self.ingestion_config.train_data_path,
				self.ingestion_config.test_data_path
			)

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)




