from src.components.data_transformation import DataTransformation

from src.exception import CustomException
from src.logger import logging

import os
import sys


class PredictPipeline:
	def __init__(self):
		pass

	def initiate_predict_pipeline(self, file_path):
		try:
			logging.info("Predict Pipeline started")

			predict_file_transform = DataTransformation()
			preprocessed_predict_file_path = predict_file_transform.initiate_data_transformation(file_path, is_pred=True)

			return preprocessed_predict_file_path

		except Exception as e:
			raise CustomException(e, sys)