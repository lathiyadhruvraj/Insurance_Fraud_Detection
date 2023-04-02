import sys
from dataclasses import dataclass
from pickle import dump, load

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
	std_scaler_pth = os.path.join('artifacts', "std_scaler.pkl")
	preprocessed_train_pth = os.path.join('artifacts', "train_preprocessed.csv")
	preprocessed_test_pth = os.path.join('artifacts', "test_preprocessed.csv")

	cols_to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
							'incident_date','incident_state','incident_city','insured_hobbies','auto_make',
							'auto_model','auto_year', 'age', 'total_claim_amount']

	custom_mapping_columns = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
	                          'property_damage', 'police_report_available', 'fraud_reported']

	dummies_columns = ['insured_occupation', 'insured_relationship', 'incident_type', 'collision_type',
                   'authorities_contacted']

	numerical_columns = ['months_as_customer', 'policy_deductable', 'umbrella_limit',
	                     'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
	                     'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim',
	                     'property_claim',
	                     'vehicle_claim']

class DataTransformation:
	def __init__(self):
		self.data_transformation_config = DataTransformationConfig()

	def drop_unnecessary_cols(self, df):
		"""
				This function is  for removing unnecessary columns
		"""
		try:

			df.drop(columns=self.data_transformation_config.cols_to_drop, inplace=True)

			return df

		except Exception as e:
			raise CustomException(e, sys)

	def replace_unknown_value_with_nan(self, df):
		"""
				This function is  for replacing unknown values with NaN

		"""
		try:
			unknown_to_replace = ['?']

			for uk in unknown_to_replace:
				df = df.replace(uk, np.nan)

			return df

		except Exception as e:
			raise CustomException(e, sys)

	def custom_mapping_for_encoding(self, df):
		"""
				This function is  for custom encoding

		"""
		try:

			# custom mapping for encoding
			df['policy_csl'] = df['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
			df['insured_education_level'] = df['insured_education_level'].map(
				{'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
			df['incident_severity'] = df['incident_severity'].map(
				{'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
			df['insured_sex'] = df['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
			df['property_damage'] = df['property_damage'].map({'NO': 0, 'YES': 1})
			df['police_report_available'] = df['police_report_available'].map({'NO': 0, 'YES': 1})
			df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})

			return df

		except Exception as e:
			raise CustomException(e, sys)

	def get_dummies_for_cat_val(self, df):
		"""
				This function is  for custom encoding

		"""
		try:

			# custom mapping for encoding
			cat_df = df.select_dtypes(include=['object']).copy()

			for col in self.data_transformation_config.dummies_columns:
				cat_df = pd.get_dummies(cat_df, columns=[col], prefix=[col], drop_first=True)

			return cat_df

		except Exception as e:
			raise CustomException(e, sys)


	def apply_standard_scaler(self, df, is_train=0):
		try:

			scaler = StandardScaler()

			num_df = df.select_dtypes(exclude=['object']).copy()
			cols = num_df.columns
			if is_train:
				scaled_df = scaler.fit_transform(num_df)
				dump(scaler, open(self.data_transformation_config.std_scaler_pth, 'wb'))
			else:
				scaler = load(open(self.data_transformation_config.std_scaler_pth, 'rb'))
				scaled_df = scaler.transform(num_df)

			scaled_df = pd.DataFrame(scaled_df, columns=cols)
			return scaled_df

		except Exception as e:
			raise CustomException(e, sys)


	def initiate_data_transformation(self, file_path, is_train=0):

		try:
			if is_train:
				logging.info("TRANSFORMATION FOR TRAIN DATA INITIATED")
			else:
				logging.info("TRANSFORMATION FOR TEST DATA INITIATED")

			logging.info("==============================================")

			df = pd.read_csv(file_path)
			logging.info("Read of data completed")
			print("original ", df.shape)

			df = self.drop_unnecessary_cols(df)
			logging.info("Deleted unnecessary columns")
			print("after drop_unnecessary_cols", df.shape)

			df = self.replace_unknown_value_with_nan(df)
			logging.info("Unknown values replaced with NaN")
			print("after replace_unknown_value_with_nan", df.shape)

			df = self.custom_mapping_for_encoding(df)
			logging.info("Custom mapping completed")
			print("after custom_mapping_for_encoding", df.shape)

			cat_df = self.get_dummies_for_cat_val(df)
			logging.info("Dummies Added")
			print("after get_dummies_for_cat_val", cat_df.shape)

			scaled_df = self.apply_standard_scaler(df, is_train)
			logging.info("Standard Scaler Completed")
			print("after apply_standard_scaler", scaled_df.shape)

			final_df = pd.concat([scaled_df, cat_df], axis=1)
			print("Final df shape", final_df.shape)

			if is_train:
				final_df.to_csv(self.data_transformation_config.preprocessed_train_pth)
				logging.info("TRANSFORMATION FOR TRAIN DATA COMPLETED")
				logging.info("=======================================")
			else:
				final_df.to_csv(self.data_transformation_config.preprocessed_test_pth)
				logging.info("TRANSFORMATION FOR TEST DATA COMPLETED")
				logging.info("=======================================")

		except Exception as e:
			raise CustomException(e, sys)

if __name__ == "__main__":
	dt = DataTransformation()
	train_path = "D:/projects/Insurance_Fraud_Detection/src/components/artifacts/train.csv"
	test_path = "D:/projects/Insurance_Fraud_Detection/src/components/artifacts/test.csv"
	dt.initiate_data_transformation(test_path, is_train=0)



# IMPLEMENT THIS PIPELINE LATER ON (ERROR: Specifying the columns using strings is only supported for pandas DataFrames)
# GO WITH NORMAL METHOD

# def get_data_transformer_object(self):
# 	"""
# 	This function is  for data transformation
#
# 	"""
# 	try:
#
# 		numerical_columns = ['months_as_customer', 'policy_deductable', 'umbrella_limit',
# 	       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
# 	       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
# 	       'vehicle_claim']
#
# 		categorical_columns = ['insured_occupation', 'insured_relationship', 'incident_type', 'collision_type',
# 			'authorities_contacted']
#
#
# 		num_pipeline = Pipeline(
# 			steps=[
# 				("scaler", StandardScaler())
# 			]
# 		)
#
# 		cat_pipeline = Pipeline(
# 			steps=[
# 				("one_hot_encoder", OneHotEncoder()),
# 				("imputer", KNNImputer(n_neighbors=3, missing_values=np.nan)),
# 			]
# 		)
#
#
# 		logging.info(f"Categorical columns: {categorical_columns}")
# 		logging.info(f"Numerical columns: {numerical_columns}")
#
# 		preprocessor = ColumnTransformer(
# 			[
# 				("num_pipeline", num_pipeline, numerical_columns),
# 				("cat_pipelines", cat_pipeline, categorical_columns)
#
# 			]
#
# 		)
#
# 		return preprocessor
#
# 	except Exception as e:
# 		raise CustomException(e, sys)
