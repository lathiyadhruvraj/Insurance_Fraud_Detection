from src.components.data_transformation import DataTransformation

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import sys
import pickle
from dataclasses import dataclass

@dataclass
class PredictPipelineConfig:
	model_path = r'D:\projects\Insurance_Fraud_Detection\artifacts\model.pkl'

class PredictPipeline:
	def __init__(self):
		self.model = PredictPipelineConfig()

	def initiate_predict_pipeline(self, file_path):
		try:
			logging.info("Predict Pipeline started")

			predict_file_transform = DataTransformation()
			preprocessed_predict_file_path = predict_file_transform.initiate_data_transformation(file_path, is_pred=True)

			loaded_model = pickle.load(open(self.model.model_path, 'rb'))
			# pred_preprocessed = os.path.join(pred_file_path, "predict_file.csv")
			pred_file = pd.read_csv(preprocessed_predict_file_path)
			pred_file.drop('fraud_reported', inplace=True, axis=1)
			result = loaded_model.predict(pred_file)

			df = pd.DataFrame(result)
			df = df.replace(0, "Fraud Not Reported")
			df = df.replace(1, "Fraud Reported")

			return df

		except Exception as e:
			raise CustomException(e, sys)

# if __name__ == "__main__":
# 	file_pth = r"D:\projects\Insurance_Fraud_Detection\artifacts\predict_files\insfd_125.csv"
# 	ob = PredictPipeline()
# 	res = ob.initiate_predict_pipeline(file_pth)
# 	print(res)