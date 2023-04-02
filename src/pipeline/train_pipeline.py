from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

import os
import sys


class TrainPipeline:
	def __init__(self):
		pass

	def initiate_train_pipeline(self):
		try:
			logging.info("Training Pipeline started")

			ingest_obj = DataIngestion()
			train_data_path, test_data_path = ingest_obj.initiate_data_ingestion()

			train_data_transform = DataTransformation()
			preprocess_train_path = train_data_transform.initiate_data_transformation(train_data_path, is_train=1)

			test_data_transform = DataTransformation()
			preprocess_test_path = test_data_transform.initiate_data_transformation(test_data_path, is_train=0)

			mt = ModelTrainer()
			roc_auc_score, model_path = mt.initiate_model_trainer(preprocess_train_path, preprocess_test_path)

			logging.info(f"Best Model roc_auc_score :-  {roc_auc_score}")

		except Exception as e:
			raise CustomException(e, sys)

if __name__ == "__main__":
	tp = TrainPipeline()
	tp.initiate_train_pipeline()





