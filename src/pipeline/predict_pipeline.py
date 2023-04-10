from src.components.data_transformation import DataTransformation

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from cfg.paths import Paths

class PredictPipeline:
	def __init__(self):
		self.paths_obj = Paths()

	def initiate_predict_pipeline(self, file_path):
		try:
			logging.info("Predict Pipeline started")
			logging.info("========================================================")

			root = Path(__file__).parent.parent.parent.resolve()
			artifacts = self.paths_obj.artifacts_dir
			model_fname = self.paths_obj.model_trainer.trained_model_fname
			model_path = root / artifacts / model_fname

			predict_file_transform = DataTransformation()
			preprocessed_predict_file_path = predict_file_transform.initiate_data_transformation(file_path, is_pred=True)

			print("Preprocessing on file finished")
			logging.info("Loading model.pkl for prediction")

			loaded_model = pickle.load(open(model_path, 'rb'))

			pred_file = pd.read_csv(preprocessed_predict_file_path)
			pred_file.drop(columns=['fraud_reported_Y'], inplace=True)
			result = loaded_model.predict(pred_file)

			logging.info("Prediction from model.pkl completed")

			df = pd.DataFrame(result)
			df = df.replace(0, "Fraud Not Reported")
			df = df.replace(1, "Fraud Reported")

			print("Predict Pipeline Completed")
			logging.info("Predict Pipeline Completed")
			return df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

# if __name__ == "__main__":
# 	file_pth = r"D:\projects\Insurance_Fraud_Detection\artifacts\predict_files\insfd_125.csv"
# 	ob = PredictPipeline()
# 	res = ob.initiate_predict_pipeline(file_pth)
# 	print(res)